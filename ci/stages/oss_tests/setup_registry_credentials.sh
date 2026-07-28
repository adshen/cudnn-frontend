#!/bin/bash
# Runs ON THE LOGIN NODE, before srun. Finds a credential pair that the GitLab
# container registry actually accepts for ENROOT_IMAGE and writes it to
# ${ENROOT_CONFIG_PATH}/.credentials for enroot/pyxis to use on the compute node.
#
# Why this exists: pyxis imports the image on the compute node, so a bad token
# surfaces there as an opaque
#     [ERROR] URL https://gitlab-master.nvidia.com/jwt/auth returned error code: 401
# with no indication of which of the several plausible causes it was. Checking
# here turns that into a specific, actionable message before any node is held.
#
# GitLab's /jwt/auth is the registry's token endpoint: it takes HTTP basic auth
# and returns a JWT whose `access` claim lists what the bearer may do. A 200
# alone is NOT proof of access -- GitLab happily returns a token with an empty
# access list for a valid identity that cannot read the repository -- so the
# granted claims are inspected too.
set -uo pipefail

: "${ENROOT_CONFIG_PATH:?ENROOT_CONFIG_PATH must be set}"
: "${ENROOT_IMAGE:?ENROOT_IMAGE must be set}"

# Parse docker://HOST[:PORT]#NAMESPACE/REPO:TAG
ref="${ENROOT_IMAGE#docker://}"
registry_host="${ref%%#*}"                 # gitlab-master.nvidia.com:5005
repo_and_tag="${ref#*#}"                   # dl/dgx/pytorch:rubin-py3-devel
repo="${repo_and_tag%:*}"                  # dl/dgx/pytorch
host="${registry_host%%:*}"                # gitlab-master.nvidia.com
auth_url="https://${host}/jwt/auth"

echo "registry preflight: repo=${repo} host=${registry_host}"

# Confirms the JWT we were handed actually grants pull on ${repo}. Falls back to
# trusting the 200 if python3 is unavailable on the login node.
claims_grant_pull() {
    local body="$1"
    command -v python3 >/dev/null 2>&1 || return 0
    python3 -c '
import base64, json, sys
body, repo = sys.argv[1], sys.argv[2]
try:
    tok = json.loads(body)["token"]
    payload = tok.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    access = json.loads(base64.urlsafe_b64decode(payload)).get("access", [])
except Exception:
    sys.exit(0)  # unparseable: do not block on a heuristic
for a in access:
    if a.get("name") == repo and "pull" in a.get("actions", []):
        sys.exit(0)
sys.exit(1)
' "$body" "$repo"
}

try_credentials() {
    local label="$1" user="$2" token="$3"
    if [ -z "${token}" ]; then
        echo "  - ${label}: skipped (no token set)"
        return 1
    fi
    if [ -z "${user}" ]; then
        echo "  - ${label}: skipped (no username)"
        return 1
    fi

    local resp code body
    resp=$(curl -sS --max-time 30 -u "${user}:${token}" \
        "${auth_url}?service=container_registry&scope=repository:${repo}:pull" \
        -w $'\n%{http_code}' 2>/dev/null)
    code=$(printf '%s' "${resp}" | tail -n1)
    body=$(printf '%s' "${resp}" | sed '$d')

    if [ "${code}" != "200" ]; then
        echo "  - ${label} (user=${user}): HTTP ${code}"
        return 1
    fi
    if ! claims_grant_pull "${body}"; then
        echo "  - ${label} (user=${user}): authenticated, but not granted pull on ${repo}"
        return 1
    fi
    echo "  - ${label} (user=${user}): OK"
    return 0
}

# Candidate credential pairs, most specific first. GitLab resolves a personal
# access token to its owner, but some deployments still match on the username,
# so the same PAT is tried under both the triggering user and the conventional
# `oauth2` placeholder before giving up.
declare -a labels users tokens
labels=(
    "RUBIN_REGISTRY_TOKEN as \$RUBIN_REGISTRY_USER"
    "RUBIN_REGISTRY_TOKEN as oauth2"
    "GITLAB_ALL_KEY as \$GITLAB_USER_LOGIN"
    "GITLAB_ALL_KEY as oauth2"
    "CI_JOB_TOKEN"
)
# RUBIN_REGISTRY_TOKEN is also tried as `oauth2` so it keeps working when the
# pipeline is triggered by someone other than the token's owner: GITLAB_USER_LOGIN
# is then a different person, and a username/token mismatch is exactly the 401
# this preflight exists to catch.
users=(
    "${RUBIN_REGISTRY_USER:-${GITLAB_USER_LOGIN:-oauth2}}"
    "oauth2"
    "${GITLAB_USER_LOGIN:-}"
    "oauth2"
    "gitlab-ci-token"
)
tokens=(
    "${RUBIN_REGISTRY_TOKEN:-}"
    "${RUBIN_REGISTRY_TOKEN:-}"
    "${GITLAB_ALL_KEY:-}"
    "${GITLAB_ALL_KEY:-}"
    "${CI_JOB_TOKEN:-}"
)

echo "trying credential pairs against ${auth_url}:"
chosen_user="" chosen_token=""
for i in "${!labels[@]}"; do
    if try_credentials "${labels[$i]}" "${users[$i]}" "${tokens[$i]}"; then
        chosen_user="${users[$i]}"
        chosen_token="${tokens[$i]}"
        break
    fi
done

if [ -z "${chosen_token}" ]; then
    cat >&2 <<EOF

FATAL: no credential pair can pull ${repo} from ${registry_host}.

This is what pyxis would have reported as a bare 401 on the compute node.

Note that dl/dgx/pytorch is an *internal*-visibility project, so ANY
authenticated gitlab-master user can read it. Group membership is therefore not
required and is not the problem -- which leaves the token itself:

  1. Most likely: the token lacks the read_registry scope. A token with only
     'api' scope (all this project needed GITLAB_ALL_KEY for until now, see
     ci/stages/notify/jobs.yml) authenticates fine against the REST API and is
     still refused by the registry. Fix: reissue GITLAB_ALL_KEY with
     read_registry added, or create a separate PAT that has it and expose it to
     this job as the masked variable RUBIN_REGISTRY_TOKEN.
  2. GITLAB_ALL_KEY is a *protected* CI/CD variable and this pipeline runs on an
     unprotected branch, so it expanded to an empty or stale value. Check the
     variable settings, or run from a protected branch.
  3. The token is expired or revoked.

Note a deploy token will NOT help here: deploy tokens are scoped to the project
that issues them, and this project cannot issue one for dl/dgx.

Fallback that avoids cross-project registry auth entirely: mirror the image into
this project's own registry and point ENROOT_IMAGE at it, where CI_JOB_TOKEN is
sufficient.
EOF
    exit 1
fi

# netrc-style, written both with and without the port so either enroot matching
# rule hits. 0600 because the login node home directory is shared.
mkdir -p "${ENROOT_CONFIG_PATH}"
umask 077
{
    printf 'machine %s login %s password %s\n' "${host}" "${chosen_user}" "${chosen_token}"
    printf 'machine %s login %s password %s\n' "${registry_host}" "${chosen_user}" "${chosen_token}"
} > "${ENROOT_CONFIG_PATH}/.credentials"
umask 022

echo "registry credentials written for user=${chosen_user} (token: ${#chosen_token} chars)"

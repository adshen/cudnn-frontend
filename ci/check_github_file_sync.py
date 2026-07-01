#!/usr/bin/env python3
"""Check that the GitLab tree stays in sync with GitHub.

GitLab is allowed to carry a hardcoded set of internal-only files. All files
that exist in GitHub must also exist in GitLab with the same mode, type, and
contents.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from dataclasses import dataclass


GITHUB_REMOTE = os.environ.get("GITHUB_SYNC_REMOTE", "https://github.com/NVIDIA/cudnn-frontend.git")
GITHUB_BRANCH = os.environ.get("GITHUB_SYNC_BRANCH", "develop")
GITHUB_REMOTE_REF = "refs/remotes/github-sync/develop"
GITLAB_REMOTE = os.environ.get("GITLAB_SYNC_REMOTE", "origin")
GITLAB_BRANCH = os.environ.get("GITLAB_SYNC_BRANCH", "develop")
GITLAB_REMOTE_REF = "refs/remotes/gitlab-sync/develop"

ALLOWED_GITLAB_ONLY_PATTERNS = (
    # Repository infrastructure and documentation.
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    "ci/**",
    "dockers/**",
    "docs/**",
    "internal/**",
    # Internal benchmark results and tests.
    "benchmark/norms/results_internal/**",
    "benchmark/sdpa_benchmark_training/results_internal/**",
    "test/pycudnnTest/**",
    # TBD GEMM feature.
    "benchmark/TBD/gemm/**",
    "python/cudnn/TBD/**",
    "test/python/TBD/gemm/**",
)


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    oid: str


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return completed.stdout


def branch_ref(branch: str) -> str:
    if branch.startswith("refs/"):
        return branch
    return f"refs/heads/{branch}"


def fetch_github_develop() -> str:
    git("fetch", "--no-tags", "--depth=1", GITHUB_REMOTE, f"+{branch_ref(GITHUB_BRANCH)}:{GITHUB_REMOTE_REF}")
    return git("rev-parse", GITHUB_REMOTE_REF).strip()


def fetch_gitlab_develop() -> str:
    git("fetch", "--no-tags", "--depth=1", GITLAB_REMOTE, f"+{branch_ref(GITLAB_BRANCH)}:{GITLAB_REMOTE_REF}")
    return git("rev-parse", GITLAB_REMOTE_REF).strip()


def tree_entries(ref: str) -> dict[str, TreeEntry]:
    entries: dict[str, TreeEntry] = {}
    for line in git("ls-tree", "-r", "--full-tree", ref).splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, oid = metadata.split()
        entries[path] = TreeEntry(mode=mode, object_type=object_type, oid=oid)
    return entries


def is_allowed_gitlab_only(path: str) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in ALLOWED_GITLAB_ONLY_PATTERNS)


def print_section(title: str, paths: list[str], *, limit: int = 100) -> None:
    print(f"\n{title}: {len(paths)}")
    for path in paths[:limit]:
        print(f"  {path}")
    if len(paths) > limit:
        print(f"  ... {len(paths) - limit} more")


def main() -> int:
    github_sha = fetch_github_develop()
    gitlab_sha = fetch_gitlab_develop()

    github_tree = tree_entries(GITHUB_REMOTE_REF)
    gitlab_tree = tree_entries(GITLAB_REMOTE_REF)

    github_paths = set(github_tree)
    gitlab_paths = set(gitlab_tree)
    shared_paths = github_paths & gitlab_paths

    github_only = sorted(github_paths - gitlab_paths)
    gitlab_only = sorted(gitlab_paths - github_paths)
    unallowed_gitlab_only = [path for path in gitlab_only if not is_allowed_gitlab_only(path)]
    allowed_gitlab_only = [path for path in gitlab_only if is_allowed_gitlab_only(path)]

    mode_type_different = sorted(path for path in shared_paths if (github_tree[path].mode, github_tree[path].object_type) != (gitlab_tree[path].mode, gitlab_tree[path].object_type))
    content_different = sorted(path for path in shared_paths if github_tree[path].oid != gitlab_tree[path].oid and path not in mode_type_different)

    print(f"GitHub {GITHUB_BRANCH}: {github_sha}")
    print(f"GitLab {GITLAB_BRANCH}: {gitlab_sha}")
    print(f"GitHub files: {len(github_tree)}")
    print(f"GitLab files: {len(gitlab_tree)}")
    print(f"Allowed GitLab-only files: {len(allowed_gitlab_only)}")

    print_section("GitHub-only files", github_only)
    print_section("Unallowed GitLab-only files", unallowed_gitlab_only)
    print_section("Shared files with different contents", content_different)
    print_section("Shared files with mode/type differences", mode_type_different)

    if github_only or unallowed_gitlab_only or content_different or mode_type_different:
        print("\nGitHub/GitLab file sync check failed.")
        print("Update GitLab to match GitHub, or update the hardcoded exception list in this script.")
        return 1

    print("\nGitHub/GitLab file sync check passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"GitHub/GitLab file sync check errored: {exc}", file=sys.stderr)
        raise SystemExit(2)

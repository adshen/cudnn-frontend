import argparse
import hashlib
import json
import platform
import re
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# should be in pytorch container
import requests

# <a*>x.y.w.z[-{hex}-{hex}]/</a> DD-month-YYYY HH:MM, [] is optional
PATTERN = re.compile(r"<a.*?>v(\d+\.\d+\.\d+\.\d+(?:-[a-f0-9]+-[a-f0-9]+)?)/</a>\s+(\d{2}-[A-Za-z]+-\d{4} \d{2}:\d{2})\s")

ARCH_MAPPING = {
    "x86_64": "x86_64",
    "aarch64": "sbsa",
}


def request_kwargs(url):
    token = os.getenv("ARTIFACTORY_TOKEN")
    user = os.getenv("ARTIFACTORY_USER")
    if not token:
        raise RuntimeError("ARTIFACTORY_TOKEN is required for Artifactory downloads")
    if user:
        return {"auth": (user, token)}
    return {"headers": {"Authorization": f"Bearer {token}"}}


def _parse_artifactory_base_url(base_url):
    """Parse base_url (e.g. https://host/artifactory/repo_key/path) into api_base, repo_key, path_in_repo."""
    parsed = urlparse(base_url)
    api_base = f"{parsed.scheme}://{parsed.netloc}/artifactory"
    path_after = parsed.path.split("/artifactory/", 1)[-1].strip("/") if "/artifactory/" in parsed.path else parsed.path.strip("/")
    path_parts = path_after.split("/")
    if not path_parts:
        raise ValueError(f"Cannot parse repo from base_url: {base_url}")
    repo_key = path_parts[0]
    path_in_repo = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
    return api_base, repo_key, path_in_repo


def _tarball_path(version, cuda_version, arch):
    """Relative path to tarball: v{version}/{cuda_version}/cudnn_debug-linux-{arch}-{version_num}.tar.gz."""
    version_num = version.split("-")[0]
    return f"v{version}/{cuda_version}/cudnn_debug-linux-{ARCH_MAPPING[arch]}-{version_num}.tar.gz"


def get_artifact_properties(base_url, version, cuda_version, arch, property_keys):
    """Get Artifactory properties for the tarball at base_url/v{version}/{cuda_version}/cudnn_debug-linux-{arch}-{version}.tar.gz."""
    api_base, repo_key, path_in_repo = _parse_artifactory_base_url(base_url)
    rel = _tarball_path(version, cuda_version, arch)
    artifact_path = f"{path_in_repo}/{rel}" if path_in_repo else rel
    url = f"{api_base}/api/storage/{repo_key}/{artifact_path}?properties={','.join(property_keys)}"
    try:
        r = requests.get(url, **request_kwargs(url), timeout=30)
        r.raise_for_status()
        return r.json().get("properties")
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


def filter_matches_by_artifact_property(matches, base_url, cuda_version, arch, cudnn_version, artifact_property_dict, max_count=3):
    """Keep only matches for which the tarball two levels down has Artifactory properties passing predicate."""
    result = []
    for m in matches:
        if len(result) >= max_count:
            break
        if not cudnn_version or m["version_num"] == cudnn_version:
            props = get_artifact_properties(base_url, m["version"], cuda_version, arch, artifact_property_dict.keys())
            if not props:
                continue
            add_match = True
            for k, v in artifact_property_dict.items():
                if k not in props or v not in props[k]:
                    add_match = False
                    break
            if add_match:
                result.append(m)
    return result


def get_artifact_sha256(base_url, version, cuda_version, arch):
    """SHA-256 of the tarball per Artifactory's storage API, or None if unavailable."""
    api_base, repo_key, path_in_repo = _parse_artifactory_base_url(base_url)
    rel = _tarball_path(version, cuda_version, arch)
    artifact_path = f"{path_in_repo}/{rel}" if path_in_repo else rel
    url = f"{api_base}/api/storage/{repo_key}/{artifact_path}"
    try:
        r = requests.get(url, **request_kwargs(url), timeout=30)
        r.raise_for_status()
        return (r.json().get("checksums") or {}).get("sha256")
    except (requests.RequestException, json.JSONDecodeError):
        return None


def sha256_of_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url, path):
    """Download url to path via a .tmp file (no partial file left on failure)."""
    tmp = Path(f"{path}.tmp")
    try:
        with requests.get(url, **request_kwargs(url), stream=True, timeout=300) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        tmp.rename(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_cudnn(base_url, cuda_version, arch, download_dir, unzip_dir, output_dir, cudnn_version=None, artifact_property_dict=None):
    response = requests.get(base_url, **request_kwargs(base_url), timeout=30)
    response.raise_for_status()
    response = response.text
    matches = PATTERN.findall(response)
    matches = [{"version": a, "version_num": a.split("-")[0], "last_modified": b} for a, b in matches]

    # sort by version
    matches = sorted(matches, key=lambda x: (tuple(map(int, x["version_num"].split("."))), x["last_modified"]), reverse=True)

    if artifact_property_dict:
        matches = filter_matches_by_artifact_property(matches, base_url, cuda_version, arch, cudnn_version, artifact_property_dict, max_count=3)
        if not matches:
            raise Exception("No version had tarball with required Artifactory properties")

    candidates = matches[:3]
    for m in candidates:
        print(m["version"], m["last_modified"])

    # if it fails to download, try a lower version one
    downloads_dir = Path(download_dir)
    downloads_dir.mkdir(exist_ok=True)
    tarball_path = None
    for match in candidates:
        version_num = match["version_num"]
        tarball_path = downloads_dir / f"cudnn-{version_num}.tar.gz"
        url = f"{base_url}/{_tarball_path(match['version'], cuda_version, arch)}"

        if tarball_path.exists():
            # A version tag can be republished with new bits, so the cached
            # tarball is only trusted when its SHA-256 matches what Artifactory
            # currently serves. On a match the multi-GB download is skipped;
            # if the checksum can't be fetched, fall back to re-downloading.
            remote_sha256 = get_artifact_sha256(base_url, match["version"], cuda_version, arch)
            if remote_sha256 and sha256_of_file(tarball_path) == remote_sha256:
                print(f"Cached tarball for {version_num} at {tarball_path} matches Artifactory SHA-256; skipping download")
                break
            print(f"Stale tarball for {version_num} at {tarball_path}; re-downloading (TOT version)")

        try:
            print(f"Fetching {version_num} from {url}")
            download_url(url, tarball_path)
            print(f"Fetching {version_num} complete")
            break
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Fetching {version_num} from {url} failed: {e}")

    if tarball_path is None or not tarball_path.exists():
        raise Exception("ERROR: Failed to get any cuDNN build")

    # Prune older cached tarballs so the CI cache stays a single file. TOT is
    # re-downloaded every run regardless, so keeping history buys nothing:
    # downloads/ just grows one multi-GB tarball per cuDNN TOT build and the
    # cache save/restore ends up taking minutes per job. The tarball we keep
    # remains the fallback if Artifactory is unreachable next run.
    for old_tarball in downloads_dir.glob("cudnn-*.tar.gz"):
        if old_tarball != tarball_path:
            print(f"Pruning cached tarball {old_tarball}")
            old_tarball.unlink()

    # extract, move, and copy
    print(f"Extracting {tarball_path}")
    subprocess.run(["tar", "xvf", str(tarball_path), "--directory", unzip_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"Moving {Path(unzip_dir) / 'cudnn'} to {output_dir}")
    subprocess.run(["mv", str(Path(unzip_dir) / "cudnn"), output_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"Copying {Path(output_dir) / 'lib'} to {Path(output_dir) / 'lib64'}")
    subprocess.run(
        ["cp", "-r", str(Path(output_dir) / "lib"), str(Path(output_dir) / "lib64")], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )

    print(f"fetch_cudnn complete")


def create_prop_dict(props):
    """Parse KEY=VALUE list into dict (last value wins for repeated keys)."""
    out = {}
    for s in props:
        k, _, v = s.partition("=")
        if not k.strip():
            raise ValueError(f"Invalid --require-artifact-prop: {s!r} (expected KEY=VALUE)")
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch cuDNN debug tarball from Artifactory, extract to /debug_cudnn, clean old cache.")
    parser.add_argument(
        "--base-url", dest="base_url", required=True, help="Base URL, e.g. https://artifactory.nvidia.com/artifactory/hw-cudnn-generic-local/CUDNN/v9.21"
    )
    parser.add_argument("--cuda-version", dest="cuda_version", required=True, help="CUDA version subdir, e.g. 13.2")
    parser.add_argument("--download-dir", dest="download_dir", default="downloads", help="Directory where downloaded tarballs are cached.")
    parser.add_argument("--unzip-dir", dest="unzip_dir", default="/", help="Directory where the tarball is extracted before moving cudnn/ into place.")
    parser.add_argument("--output-dir", dest="output_dir", default="/debug_cudnn", help="Directory where the extracted cudnn/ tree will be moved.")
    parser.add_argument("--cudnn-version", dest="cudnn_version", help="Optional cuDNN version x.y.w.z")
    parser.add_argument("--arch", dest="arch", default=platform.machine(), choices=sorted(ARCH_MAPPING), help="Which arch's tarball to fetch. Defaults to the current machine.")
    parser.add_argument(
        "--require-artifact-prop",
        action="append",
        metavar="KEY=VALUE",
        help="Require Artifactory property (repeat for multiple). E.g. --require-artifact-prop pipeline_type=merge_train --require-artifact-prop arch=x86_64",
    )
    args = parser.parse_args()

    if args.arch not in ARCH_MAPPING:
        raise SystemExit(f"unsupported arch {args.arch!r}; pass --arch with one of {sorted(ARCH_MAPPING)}")

    prop_dict = None

    if args.require_artifact_prop:
        prop_dict = create_prop_dict(args.require_artifact_prop)

    fetch_cudnn(args.base_url, args.cuda_version, args.arch, args.download_dir, args.unzip_dir, args.output_dir, args.cudnn_version, artifact_property_dict=prop_dict)

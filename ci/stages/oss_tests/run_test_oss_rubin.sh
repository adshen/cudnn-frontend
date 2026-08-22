#!/bin/bash
# Runs INSIDE the Rubin container on a hecate compute node, launched by
# `srun --container-image` from ci/stages/oss_tests/jobs.yml (oss:rubin).
#
# Differs from run_test_oss.sh (docker-runner path) in three ways:
#   1. cuDNN comes from the container, not from ci/common/fetch_cudnn.py, so
#      there is no /debug_cudnn to put on LD_LIBRARY_PATH.
#   2. cudnn_frontend is pip-installed rather than imported from build/, so
#      PYTHONPATH is left alone.
#   3. cutedsl is the internal (unreleased) nvidia-cutlass-dsl-internal wheel
#      from urm, which is what carries the Rubin/sm107 kernels.
set -exo pipefail

RESULTS_DIR="/workspace/${RESULTS_DIR_NAME:?RESULTS_DIR_NAME must be set}"
mkdir -p "${RESULTS_DIR}"
# The container runs as root while the login-node user owns the build dir;
# make results readable/cleanable from the runner side either way.
trap 'chmod -R 777 "${RESULTS_DIR}" 2>/dev/null || true' EXIT

exec > >(tee "${RESULTS_DIR}/oss-rubin.log") 2>&1

echo "=== $(hostname) ($(uname -m)) ==="
nvidia-smi

visible=$(nvidia-smi -L | wc -l)
if [ "${visible}" -lt "${GPUS_PER_NODE:?GPUS_PER_NODE must be set}" ]; then
    echo "FATAL: ${visible} GPUs visible, expected >= ${GPUS_PER_NODE}"
    exit 1
fi

# --- Rubin gate -------------------------------------------------------------
# The fe_api tests self-skip below compute capability 100, so on non-Rubin
# hardware this job would "pass" while testing nothing. Fail loudly instead.
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')
echo "compute capability: ${compute_cap} (expected ${EXPECTED_COMPUTE_CAP:?EXPECTED_COMPUTE_CAP must be set})"
if [ "${compute_cap}" != "${EXPECTED_COMPUTE_CAP}" ]; then
    echo "FATAL: allocated a non-Rubin node (sm${compute_cap}); check the SLURM partition/constraint."
    exit 1
fi

# Build outside the shared mount: the workspace is the runner's clone on a
# shared filesystem and must not accumulate in-tree build artifacts. Use a tar
# pipe instead of cp -a — the mount does not support preserving permissions or
# xattrs into the container filesystem ("Operation not supported").
mkdir -p /tmp/cudnn_frontend_src
tar -C /workspace --exclude="./${RESULTS_DIR_NAME}" --exclude=./.git -cf - . \
    | tar -C /tmp/cudnn_frontend_src -xf -
cd /tmp/cudnn_frontend_src

# --- install ----------------------------------------------------------------
# The Rubin image is a devel PyTorch base, not a CI image: it has no pytest.
# requirements.txt is the repo's own list and carries both pytest and
# pytest-xdist, the latter being required for the -n sharding used below.
pip install --no-cache-dir -r requirements.txt

# Prefer --no-build-isolation (compute nodes may lack pypi egress); the DLFW
# image ships setuptools/cmake/ninja/pybind11. Fall back to an isolated build.
pip install -v --no-cache-dir --no-build-isolation .[cutedsl] \
    || pip install -v --no-cache-dir .[cutedsl]

# Swap the pyproject-resolved cutedsl for the version this job pins
# (CUTLASS_DSL_PACKAGE, currently the public 4.8 prerelease — the first public
# line whose Arch enum carries sm_107a; 4.7.0 stops at sm_103a).
# Install-then-replace, because the extra is what pulls in the rest of the
# cutedsl dependency set (cuda-python, apache-tvm-ffi, torch-c-dlpack-ext), so
# uninstalling the DSL afterwards is cheaper than hand-listing those deps and
# letting them drift from pyproject.toml.
#
# The Rubin image ships nvidia-cutlass-dsl-internal preinstalled. All of these
# wheels unpack into the same nvidia_cutlass_dsl/dsl_packages/cutlass/
# directory, so whichever dist is installed last silently owns the files while
# the others still look "already satisfied" to pip. To make the pinned wheel
# unambiguously the one on disk: remove the pyproject-resolved public dist
# *and its libs* packages *and* the image's internal dist, then install the
# pinned package fresh (with deps — the public wheels split the compiled
# pieces into nvidia-cutlass-dsl-libs-* which --no-deps would skip).
pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-internal \
    nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core \
    nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 || true
pip install --no-cache-dir \
    "${CUTLASS_DSL_PACKAGE:?CUTLASS_DSL_PACKAGE must be set}" \
    --extra-index-url "${CUTLASS_DSL_INDEX_URL:?CUTLASS_DSL_INDEX_URL must be set}"

# Prove the DSL on the path actually understands Rubin before spending an hour of
# node time on it. Without this the failure mode is 3000 identical KeyErrors deep
# in collection, which reads as "the change broke everything" rather than "the
# wrong wheel won".
python - <<'PY'
import sys
import importlib.metadata as md

for d in sorted(md.distributions(), key=lambda x: (x.metadata["Name"] or "").lower()):
    name = d.metadata["Name"] or ""
    if "cutlass" in name.lower():
        print(f"  dist: {name} == {d.version}")

import cutlass
print("  cutlass.__file__   :", cutlass.__file__)
print("  cutlass.__version__:", getattr(cutlass, "__version__", "unknown"))

from cutlass.base_dsl.arch import Arch

members = list(Arch.__members__)
print("  Arch sm_10x members:", [m for m in members if m.startswith("sm_10")])
if "sm_107a" not in members:
    sys.exit(
        "FATAL: the resolved cutlass DSL has no sm_107a in its Arch enum, so every "
        "Rubin test would fail with KeyError: 'sm_107a'. The public "
        "nvidia-cutlass-dsl has shadowed nvidia-cutlass-dsl-internal."
    )
print("  sm_107a: OK")
PY

# Fail here rather than midway through collection. requirements.txt pins
# numpy<2.0.0, which can downgrade the numpy the container's torch was built
# against, so torch is re-imported explicitly after the install.
python -c 'import pytest, xdist; print("pytest:", pytest.__version__)'
python -c 'import numpy, torch; print("numpy:", numpy.__version__); print("torch:", torch.__version__)'
python -c 'import cutlass; print("cutlass dsl:", getattr(cutlass, "__version__", "unknown"))'
python -c 'import cudnn; print("cudnn frontend:", cudnn.__version__); print("cudnn backend:", cudnn.backend_version_string())'

# --- test -------------------------------------------------------------------
# Write the junit report straight to the mounted results dir so it survives even
# if a later step in this script dies.
# PYTEST_TARGETS narrows the run to a subset (space-separated paths / node ids).
# The full fe_api suite does not fit in SRUN_TIMELIMIT on Rubin -- it reached 92%
# at the 100 minute mark -- so a targeted subset is the only way to get a junit
# report out of a single allocation.
read -r -a pytest_targets <<< "${PYTEST_TARGETS:-test/python/fe_api}"
echo "pytest targets: ${pytest_targets[*]}"
set +e
pytest -n "${PYTEST_WORKERS:-4}" \
       --junit-xml="${RESULTS_DIR}/result-junit.xml" \
       --no-header --tb=short \
       "${pytest_targets[@]}"
pytest_status=$?
set -e

touch "${RESULTS_DIR}/complete.ok"
echo "=== oss:rubin finished (pytest exit ${pytest_status}) ==="
exit "${pytest_status}"

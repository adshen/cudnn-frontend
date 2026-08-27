#!/bin/bash
# Runs INSIDE the container on a SLURM compute node, launched by
# `srun --container-image` from ci/stages/frost_tests/jobs.yml (the frost jobs
# tagged hecate / lyris).
#
# Differs from run_frost_tests.sh (docker-runner path) in three ways:
#   1. ci/common/fetch_cudnn.py runs here rather than in a before_script,
#      because it has to run on the compute node inside the container.
#   2. cudnn_frontend is pip-installed from the mounted source rather than
#      imported from build/, so PYTHONPATH is left alone.
#   3. The results dir is a mounted host path, because the job that reads the
#      junit report runs back on the login node.
set -exo pipefail

RESULTS_DIR="/workspace/${RESULTS_DIR_NAME:?RESULTS_DIR_NAME must be set}"
mkdir -p "${RESULTS_DIR}"
# The container runs as root while the login-node user owns the build dir;
# make results readable/cleanable from the runner side either way.
trap 'chmod -R 777 "${RESULTS_DIR}" 2>/dev/null || true' EXIT

exec > >(tee "${RESULTS_DIR}/frost-cluster.log") 2>&1

echo "=== $(hostname) ($(uname -m)) ==="
nvidia-smi

# --- arch gate --------------------------------------------------------------
# The FROST tests self-skip off the architecture their engines target, so on a
# wrongly allocated node this job would "pass" while testing nothing. The whole
# point of these jobs is the hardware, so fail loudly instead.
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')
echo "compute capability: ${compute_cap} (expected ${EXPECTED_COMPUTE_CAP:?EXPECTED_COMPUTE_CAP must be set})"
if [ "${compute_cap}" != "${EXPECTED_COMPUTE_CAP}" ]; then
    echo "FATAL: allocated an sm${compute_cap} node; check the SLURM partition/constraint."
    exit 1
fi

# Build outside the shared mount: the workspace is the runner's clone on a
# shared filesystem and must not accumulate in-tree build artifacts. Use a tar
# pipe instead of cp -a -- the mount does not support preserving permissions or
# xattrs into the container filesystem ("Operation not supported").
mkdir -p /tmp/cudnn_frontend_src
tar -C /workspace --exclude="./${RESULTS_DIR_NAME}" --exclude=./.git -cf - . \
    | tar -C /tmp/cudnn_frontend_src -xf -
cd /tmp/cudnn_frontend_src

python3 ci/common/fetch_cudnn.py \
    --base-url "${CUDNN_FETCH_BASE_URL:?CUDNN_FETCH_BASE_URL must be set}" \
    --cuda-version "${CUDNN_FETCH_CUDA_VERSION:?CUDNN_FETCH_CUDA_VERSION must be set}" \
    --require-artifact-prop pipeline_type=schedule
export CUDNN_PATH=/debug_cudnn
export LD_LIBRARY_PATH=/debug_cudnn/lib:${LD_LIBRARY_PATH}

# --- install ----------------------------------------------------------------
# These are devel PyTorch images, not CI images: they have no pytest.
# requirements.txt is the repo's own list and carries both pytest and
# pytest-xdist, the latter being required for the -n sharding used below.
pip install --no-cache-dir -r requirements.txt

# Prefer --no-build-isolation (compute nodes may lack pypi egress); the DLFW
# image ships setuptools/cmake/ninja/pybind11. Fall back to an isolated build.
pip install -v --no-cache-dir --no-build-isolation .[cutedsl] \
    || pip install -v --no-cache-dir .[cutedsl]

# Pin the DSL the job asked for, replacing both whatever .[cutedsl] resolved and
# whatever the image ships. FROST_ prefix rather than the bare
# CUTLASS_DSL_VERSION: the DLFW images export one of their own, which is how the
# docker-runner jobs silently ended up on 4.4.2 (see run_frost_tests.sh).
#
# The pin is applied with --no-deps, and the libs packages are named explicitly
# because --no-deps means the [cu13] extra pulls nothing:
#
#   * a prerelease pin (4.8.0a0) does not satisfy pyproject's
#     `nvidia-cutlass-dsl>=4.5.0` for pip's default resolver, so `.[cutedsl]`
#     above happily *downgrades* the Rubin image's preinstalled 4.8 to the
#     public 4.7.0 and this step has to put it back;
#   * resolving deps here re-opens the image's own pip constraint file, which
#     pins cuda-python to a dev build (13.4.2.dev6) that no cutlass-dsl release
#     declares support for -- the resolution the Rubin job died on
#     (ResolutionImpossible, "The user requested (constraint) cuda-python==...").
#
# cuda-python, apache-tvm-ffi and the rest of the cutedsl dependency set are
# already installed by the .[cutedsl] step above, so nothing is lost.
dsl_pin="${FROST_CUTLASS_DSL_VERSION:?FROST_CUTLASS_DSL_VERSION must be set}"
pip install --no-cache-dir --force-reinstall --no-deps \
    "nvidia-cutlass-dsl==${dsl_pin}" \
    "nvidia-cutlass-dsl-libs-base==${dsl_pin}" \
    "nvidia-cutlass-dsl-libs-core==${dsl_pin}" \
    "nvidia-cutlass-dsl-libs-cu13==${dsl_pin}" \
    ${FROST_CUTLASS_DSL_INDEX_URL:+--extra-index-url "${FROST_CUTLASS_DSL_INDEX_URL}"}

# A wheel other than the pinned one means the suite is testing something else.
python -c "import importlib.metadata as md; v = md.version('nvidia-cutlass-dsl'); assert v.startswith('${dsl_pin}'), f'cutlass-dsl {v} installed, expected ${dsl_pin}'"

# Prove the DSL understands this GPU before spending an allocation on it.
# Without this the failure mode is thousands of identical KeyErrors deep in
# collection, which reads as "the change broke everything".
EXPECTED_ARCH="sm_${EXPECTED_COMPUTE_CAP}a" python - <<'PY'
import os
import sys

import cutlass
from cutlass.base_dsl.arch import Arch

want = os.environ["EXPECTED_ARCH"]
print("  cutlass.__file__   :", cutlass.__file__)
print("  cutlass.__version__:", getattr(cutlass, "__version__", "unknown"))
print("  Arch sm_1xx members:", [m for m in Arch.__members__ if m.startswith("sm_1")])
if want not in Arch.__members__:
    sys.exit(f"FATAL: the resolved cutlass DSL has no {want} in its Arch enum.")
print(f"  {want}: OK")
PY

# The FROST tvm-ffi front door (~4.3x lower host dispatch) degrades silently if
# tvm_ffi is missing, so a lost dependency would hide as a green build.
python -c "import tvm_ffi; print('tvm_ffi', tvm_ffi.__version__)"
python -c 'import pytest, xdist; print("pytest:", pytest.__version__)'
cudnn_fetched=$(ls /debug_cudnn/lib/libcudnn.so.*.*.* | sed 's/.*libcudnn\.so\.//')
CUDNN_FETCHED="${cudnn_fetched}" python -c '
import os, cudnn
print("cudnn frontend:", cudnn.__version__)
print("cudnn backend:", cudnn.backend_version_string(), "(fetched", os.environ["CUDNN_FETCHED"] + ")")
assert os.environ["CUDNN_FETCHED"].startswith(cudnn.backend_version_string()), "loaded the image cuDNN, not the fetched one"
'

# --- test -------------------------------------------------------------------
# The FROST engines are opt-in manifest rows (cudnn/engines/manifest.py): off by
# default until they have the arch coverage to serve graphs unasked.
export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1

read -r -a frost_test_paths <<< "${FROST_TEST_PATHS:?FROST_TEST_PATHS must be set}"
echo "pytest targets: ${frost_test_paths[*]}"
set +e
pytest -n "${PYTEST_WORKERS:-4}" \
       --junit-xml="${RESULTS_DIR}/result-junit.xml" \
       --no-header --tb=short \
       "${frost_test_paths[@]}"
pytest_status=$?
set -e

# The login-node job treats a missing marker as failure, so a payload that dies
# before pytest runs cannot look green.
touch "${RESULTS_DIR}/complete.ok"
exit "${pytest_status}"

#!/bin/bash
# Runs INSIDE the container, once per allocated node (srun --ntasks-per-node=1).
# Verifies GPU visibility, installs cudnn_frontend from the mounted source, and
# executes a small cudnn graph on every visible GPU (ci/stages/multi_gpu_tests/multigpu_dist_test.py).
set -exo pipefail

NODE_TAG="node$(printf '%02d' "${SLURM_NODEID:-0}")"
RESULTS_DIR="/workspace/multigpu-results"
mkdir -p "${RESULTS_DIR}"
# The container runs as root while the login-node user owns the build dir;
# make results cleanable/readable either way.
trap 'chmod -R 777 "${RESULTS_DIR}" 2>/dev/null || true' EXIT

exec > >(tee "${RESULTS_DIR}/smoke-${NODE_TAG}.log") 2>&1

echo "=== ${NODE_TAG}: $(hostname) ($(uname -m)) ==="
nvidia-smi
visible=$(nvidia-smi -L | wc -l)
if [ "${visible}" -lt "${GPUS_PER_NODE:?GPUS_PER_NODE must be set}" ]; then
    echo "FATAL: ${visible} GPUs visible on ${NODE_TAG}, expected >= ${GPUS_PER_NODE}"
    exit 1
fi

# Build outside the shared mount: multiple nodes run concurrently and must not
# race on in-tree build artifacts. Use a tar pipe instead of cp -a — the
# workspace mount does not support preserving permissions/xattrs into the
# container filesystem (cp -a fails with "Operation not supported").
mkdir -p /tmp/cudnn_frontend_src
tar -C /workspace --exclude=./multigpu-results --exclude=./.git -cf - . \
    | tar -C /tmp/cudnn_frontend_src -xf -
cd /tmp/cudnn_frontend_src

# Prefer no-build-isolation (compute nodes may lack pypi egress); the DLFW
# image ships setuptools/cmake/ninja/pybind11. Fall back to an isolated build.
pip install --no-cache-dir --no-build-isolation . || pip install --no-cache-dir .

# Distributed smoke: one rank per GPU on this node, NCCL all-reduce + a cudnn
# graph per rank. --standalone = node-local rendezvous (each allocated node
# runs its own torchrun; cross-node rendezvous comes with real multi-node tests).
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" ci/stages/multi_gpu_tests/multigpu_dist_test.py

touch "${RESULTS_DIR}/complete-${NODE_TAG}.ok"
echo "=== ${NODE_TAG}: OK ==="

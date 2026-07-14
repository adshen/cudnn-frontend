#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo "Installed cuda version" $CUDA_VERSION_
}

function install_deps() {
    pip install nvidia-cutlass-dsl-internal --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi-local/simple
}

function run_frost_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64
    export NV_CUDNN_FE_ENABLE_FROST_ENGINES=1
    
    echo "Running frost tests."
    pytest test/python/frost/gemm test/python/frost/sdpa -n 4 --junit-xml=result-junit.xml --no-header --tb=short
}

display_header
install_deps
run_frost_tests

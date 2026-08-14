#!/bin/bash
set -e

# Usage: run_frost_tests.sh [pytest path]...
# The CI splits the suite one job per opset folder (see jobs.yml) and passes that
# opset's paths here; with no arguments the whole FROST suite runs, which is what
# you want when invoking this by hand.
FROST_TEST_PATHS=("$@")
if [ ${#FROST_TEST_PATHS[@]} -eq 0 ]; then
    FROST_TEST_PATHS=(
        test/python/gemm/frost
        test/python/sdpa/frost
        test/python/linear_attention
        test/python/test_mhas_v2.py
    )
fi

function display_header() {
    nvidia-smi
    echo "Installed cuda version" $CUDA_VERSION_
}

function install_deps() {
    # [cu13] to match pyproject.toml and the CUDA 13.x container: without the
    # extra, pip resolves nvidia-cutlass-dsl-libs-cu12 and the DSL runs a
    # CUDA-12 libNVVM (statically linked into _cutlass_ir.cu12.so) against a
    # CUDA 13 toolkit.
    #
    # 4.7.0 is a PyPI release now, so this pins the public wheel instead of the
    # 4.7.0a0 prerelease from the urm internal index -- both packages are on
    # PyPI, so no --extra-index-url is needed.
    pip install "nvidia-cutlass-dsl[cu13]==4.7.0" "apache-tvm-ffi>=0.1.11"
    # The FROST tvm-ffi front door (~4.3x lower host dispatch) degrades silently
    # if tvm_ffi is missing, so a lost dependency would hide as a green build.
    python -c "import tvm_ffi; print('tvm_ffi', tvm_ffi.__version__)"
    # Record what actually resolved, so a failure can be attributed to a build.
    pip list | grep -i cutlass
}

function run_frost_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64

    # The FROST engines are opt-in manifest rows (cudnn/engines/manifest.py):
    # off by default until they have the arch coverage to serve graphs
    # unasked. This job is what exercises them.
    export CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1

    echo "Running frost tests: ${FROST_TEST_PATHS[*]}"
    # test_mhas_v2 runs through cudnn.pygraph; with the opt-in set the FROST
    # engines join the plan list for every graph they claim, ranked against the
    # backend's own plans by cudnn/engines/heuristics.py.
    # The run prints a 'FROST routing' summary tallying frost-vs-native graphs.
    pytest "${FROST_TEST_PATHS[@]}" -n 4 --junit-xml=result-junit.xml --no-header --tb=short
}

display_header
install_deps
run_frost_tests

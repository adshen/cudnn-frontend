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
    # 4.7.0 is a PyPI release now, so the default pins the public wheel instead
    # of the 4.7.0a0 prerelease from the urm internal index -- both packages are
    # on PyPI, so no --extra-index-url is needed for it.
    #
    # FROST_CUTLASS_DSL_VERSION / FROST_CUTLASS_DSL_INDEX_URL let a job run the
    # suite against an unreleased wheel -- the nightly frost:cutlass-dsl-4.8
    # jobs in jobs.yml pin the 4.8 branch tip off the urm internal index. Leave
    # the index empty for a public version: urm also carries local-version
    # builds of the released wheels (4.7.0+<date>.<sha>), and pip prefers those
    # over PyPI's 4.7.0.
    #
    # The FROST_ prefix is not decoration. The NGC PyTorch image these jobs run
    # in exports its own CUTLASS_DSL_VERSION (4.4.2 in cudnn_13.3.0, stale even
    # against the 4.5.2 the image ships), so reading that bare name here
    # installed 4.4.2 instead of the pin and failed the suite on `No module
    # named cutlass.experimental`.
    FROST_CUTLASS_DSL_VERSION="${FROST_CUTLASS_DSL_VERSION:-4.7.0}"
    pip install "nvidia-cutlass-dsl[cu13]==${FROST_CUTLASS_DSL_VERSION}" \
        ${FROST_CUTLASS_DSL_INDEX_URL:+--extra-index-url "${FROST_CUTLASS_DSL_INDEX_URL}"} \
        "apache-tvm-ffi>=0.1.11"
    # A wheel other than the pinned one is a broken job, not 51 mystery failures.
    python -c "import importlib.metadata as md; v = md.version('nvidia-cutlass-dsl'); assert v.startswith('${FROST_CUTLASS_DSL_VERSION}'), f'cutlass-dsl {v} installed, expected ${FROST_CUTLASS_DSL_VERSION}'"
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

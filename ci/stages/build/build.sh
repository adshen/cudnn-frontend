#!/bin/bash

# MR pipelines never execute the C++ samples/tests binaries from this build:
# every downstream consumer of build:dev / build:rel artifacts only needs the
# python bindings (PYTHONPATH=build), bin/tests runs from the separate
# sanitizer build, and compile coverage for samples/tests is already provided
# by the analysis:* and build:win jobs. Only the scheduled qa_matrix pipeline
# runs bin/samples + bin/legacy_samples, so its build jobs opt back in by
# exporting FE_CI_BUILD_SAMPLES_AND_TESTS=ON (see ci/qa_matrix/jobs.yml).
FE_CI_BUILD_SAMPLES_AND_TESTS="${FE_CI_BUILD_SAMPLES_AND_TESTS:-OFF}"

function build_commands() {
    mkdir build
    cd build
    cmake -DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON \
          -DCUDNN_FRONTEND_BUILD_SAMPLES="${FE_CI_BUILD_SAMPLES_AND_TESTS}" \
          -DCUDNN_FRONTEND_BUILD_TESTS="${FE_CI_BUILD_SAMPLES_AND_TESTS}" \
          ../
    cmake --build . -j16
}

build_commands

#!/bin/bash
set -e

function build_commands() {
    mkdir build
    cd build
    # -Wno-error flags suppress false positives from GCC 13 in third-party
    # code (Catch2 regex, pybind11 memmove) that get promoted by -Werror.
    cmake \
        -DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=OFF \
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -Wno-error=maybe-uninitialized -Wno-error=array-bounds" \
        -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined" \
        ../
    cmake --build . -j16
}

build_commands

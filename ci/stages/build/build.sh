#!/bin/bash

function build_commands() {
    mkdir build
    cd build
    cmake -DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON ../
    cmake --build . -j16
}

build_commands

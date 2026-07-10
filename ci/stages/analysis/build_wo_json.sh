#!/bin/bash

set -x

function build_wo_json_commands() {
    mkdir build
    cd build
    cmake -DCUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON -DCUDNN_FRONTEND_SKIP_JSON_LIB=ON ../
    c++ -DCUDNN_FRONTEND_SKIP_JSON_LIB -I../include -I_deps/catch2-src/src/catch2/.. -I_deps/catch2-build/generated-includes -I /usr/local/cuda/targets/x86_64-linux/include -std=gnu++17 -Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function -o ./validate.cpp.o -c ../test/cpp/validate.cpp

}    

build_wo_json_commands

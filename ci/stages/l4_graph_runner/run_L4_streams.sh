#!/bin/bash
set -e

input_file="./ci/stages/l4_graph_runner/L4_testlist.txt"
total_lines=$(wc -l < "$input_file")
stream_group_size=16

function display_header() {
    nvidia-smi
    echo "Installed cudnn version" $CUDNN_VERSION_
    echo "Installed cuda version" $CUDA_VERSION_
}

function run_python_streams() {
    export PYTHONPATH=build

    
    # This command runs tests in batches of $stream_group_size, and will stop running new batches when a test failed.
    # for ((i = 1; i <= total_lines; i += stream_group_size)); do
    #     echo "Running tests $i - $stream_group_size out of $total_lines"
    #     ./test/pycudnnTest.py -RgrStream --stream_start "$i" --stream_group_size "$stream_group_size" < "$input_file"
    #done
    # This command runs a single process for all tests (this ensures all tests are run even if errors are encountered)
    ./test/pycudnnTest/pycudnnTest.py -RgrStream < "$input_file"
}

display_header
run_python_streams
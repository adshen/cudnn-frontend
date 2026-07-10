#!/bin/bash
set -e

function display_header() {
    nvidia-smi
}

function run_python_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64
    
    pytest -n 4 --junit-xml=result-junit.xml --tb=short test/python/test_mhas.py
}

display_header
run_python_tests
#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo "Installed cudnn version" $CUDNN_VERSION_
    echo "Installed cuda version" $CUDA_VERSION_
}

function run_python_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=/debug_cudnn/lib64/
    
    # Graph tests from json
    python3 test/pycudnnTest/pycudnnTest.py --testPath test/pycudnnTest/json_graph_defs/graphTests.json --testName ConvRelu1
    # Python defined graph tests
    python3 test/pycudnnTest/pycudnnTest.py --testPath test/pycudnnTest/python_graph_defs/basic_tests.py
    # Legacy graphRunner test with reference
    python3 test/pycudnnTest/pycudnnTest.py -Dforce_jit_dbg= 1 -R graphRunner -backendEngine -3 -jsonTestName= ConvRelu_abstract1 -x  -Pcomp h -Pin h -Pout h -convStrideA 1,1 -dilationA 1,1 -dimA 4,32,32,32 -filtA 32,32,3,3 -dimOut 4,32,30,30 -formatAll 1 -padA 0,0 -atol 3.0e-03 -rtol 2.5 -minDevVer 750 -S  -b  -serialization 0 -gpuRef
    # Legacy graphRunner test with timing loop
    python3 test/pycudnnTest/pycudnnTest.py -Dforce_jit_dbg= 1 -R graphRunner -backendEngine -3 -jsonTestName= ConvRelu_abstract1 -x  -Pcomp h -Pin h -Pout h -convStrideA 1,1 -dilationA 1,1 -dimA 4,32,32,32 -filtA 32,32,3,3 -dimOut 4,32,30,30 -formatAll 1 -padA 0,0 -atol 3.0e-03 -rtol 2.5 -minDevVer 750 -S  -b  -serialization 0 -gpuRef -T10
}

display_header
run_python_tests

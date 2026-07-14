#!/bin/bash
set -e

function display_header() {
    nvidia-smi
}

function run_python_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64

    local oss_tests_shard1=(
        dglu
        dswiglu
        dsrelu
        wgrad
        backward
        sdpa_bwd
        rmsnorm_rht_amax
    )
    local oss_tests_shard2=(
        # Add shard 2 filters here.
    )
    local oss_tests_shard3=(
        # Add shard 3 filters here.
    )
    local sharded_tests=(
        "${oss_tests_shard1[@]}"
        "${oss_tests_shard2[@]}"
        "${oss_tests_shard3[@]}"
    )

    local test_filter=()
    local shard_filter
    if [[ "${OSS_TEST_SHARD:-}" == "shard0" ]]; then
        printf -v shard_filter "%s or " "${sharded_tests[@]}"
        test_filter=(-k "not (${shard_filter% or })")
    elif [[ "${OSS_TEST_SHARD:-}" == "shard1" ]]; then
        printf -v shard_filter "%s or " "${oss_tests_shard1[@]}"
        test_filter=(-k "${shard_filter% or }")
    elif [[ "${OSS_TEST_SHARD:-}" == "shard2" ]]; then
        printf -v shard_filter "%s or " "${oss_tests_shard2[@]}"
        test_filter=(-k "${shard_filter% or }")
    elif [[ "${OSS_TEST_SHARD:-}" == "shard3" ]]; then
        printf -v shard_filter "%s or " "${oss_tests_shard3[@]}"
        test_filter=(-k "${shard_filter% or }")
    fi

    pytest -n 4 --junit-xml=result-junit.xml --no-header --tb=short "${test_filter[@]}" test/python/fe_api
}

display_header
run_python_tests

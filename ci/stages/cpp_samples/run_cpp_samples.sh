#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo "Installed cudnn version" $CUDNN_VERSION_
    echo "Installed cuda version" $CUDA_VERSION_
}

function concat_xml() {

    local file1="$1"
    local file2="$2"
    local output="$3"

    # Create temporary files
    local temp1=$(mktemp)
    local temp2=$(mktemp)

    # Remove last line from first file (assumed to be closing root tag)
    sed '$d' "$file1" > "$temp1"

    # Remove XML declaration and root tag from second file
    # Skip first line (XML declaration) and root tag
    tail -n +2 "$file2" | sed '1d' > "$temp2"

    # Combine files
    cat "$temp1" "$temp2" > "$output"

    # Clean up temporary files
    rm "$temp1" "$temp2"

    echo "Successfully concatenated XML files to $output"
}

function run_cpp_samples() {
    build/bin/samples --reporter JUnit::out=result-junit0.xml --reporter console::out=-::colour-mode=ansi
    build/bin/legacy_samples --reporter JUnit::out=result-junit1.xml --reporter console::out=-::colour-mode=ansi
    concat_xml result-junit0.xml result-junit1.xml result-junit.xml
}

display_header
run_cpp_samples

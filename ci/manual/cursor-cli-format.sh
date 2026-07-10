#!/bin/bash
#
# cursor-agent installation:
#   curl https://cursor.com/install -fsSL | bash
#
# Usage:
#   ./ci/manual/cursor-cli-format.sh ci/manual/cursor-cli-format-py.txt
#

if [ -z "$1" ]; then
    echo "Usage: $0 <prompt_file>"
    exit 1
fi

# black python/cudnn/ --exclude "(gemm_amax|native_sparse_attention|gemm_swiglu)" --line-length 160


# isort python/cudnn/ \
#     --skip gemm_amax \
#     --skip native_sparse_attention \
#     --skip gemm_swiglu

cursor-agent --model "gemini-3-flash" -p "$(cat "$1")"

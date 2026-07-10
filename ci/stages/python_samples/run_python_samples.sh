#!/bin/bash
set -e

GPU_CC=`nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | cut -d"." -f1`
CUDNN_BE_VERSION=`python3 -c "import cudnn;print(cudnn.backend_version())"`

export LD_LIBRARY_PATH=/debug_cudnn/lib64

if [ "${GPU_CC}" -ge "9" ]; then
    jupyter execute samples/python/0*
    jupyter execute samples/python/2[0-2]*
    jupyter execute samples/python/29*
    jupyter execute samples/python/3[0-2]*
    jupyter execute samples/python/50*
    jupyter execute samples/python/51*
else
    jupyter execute samples/python/00*
    jupyter execute samples/python/02*
    jupyter execute samples/python/2[0-2]*
    jupyter execute samples/python/29*
    jupyter execute samples/python/3[0-2]*
    jupyter execute samples/python/50*
    jupyter execute samples/python/51*
fi

# Run paged attention for SM>=80 and cuDNN >= 9.5
if [ "${CUDNN_BE_VERSION}" -ge "90500" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/52*
fi

# Run paged attention with packed K/V block tables for SM>=80 and cuDNN >= 9.11
if [ "${CUDNN_BE_VERSION}" -ge "91100" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/53*
fi

# Run adaptive layer norm for SM>=80 and cuDNN >= 9.9
if [ "${CUDNN_BE_VERSION}" -ge "90900" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/27*
    jupyter execute samples/python/28*
fi

# Run layer norm zero-centered gamma for SM>=80 and cuDNN >= 9.10
if [ "${CUDNN_BE_VERSION}" -ge "91000" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/24*
    jupyter execute samples/python/25*
fi

# Run layer norm with relu bitmask fusion for SM>=80 and cuDNN >= 9.13
if [ "${CUDNN_BE_VERSION}" -ge "91300" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/26*
fi

# Run layer norm with pointwise add fusion for SM>=80 and cuDNN >= 9.14
if [ "${CUDNN_BE_VERSION}" -ge "91400" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/23*
fi

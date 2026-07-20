ARG DLFW_MONTH

FROM nvcr.io/nvidia/pytorch:${DLFW_MONTH}-py3

ARG CUDNN_VERSION_
ARG CUDA_VERSION_
ARG SKIP_CUDNN=false


RUN echo "CUDNN_VERSION is: $CUDNN_VERSION_"
RUN echo "CUDA_VERSION is: $CUDA_VERSION_"

ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    CUDNN_VERSION_=${CUDNN_VERSION_} \
    CUDA_VERSION_=${CUDA_VERSION_}

SHELL ["/bin/bash", "-c"]

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update -qq --fix-missing

RUN apt-get -y install cuda-toolkit-${CUDA_VERSION_}

RUN echo 0 | update-alternatives --config cuda

RUN CUDNN_VERSION_M=${CUDNN_VERSION_%%.*} && \
    CUDA_VERSION_M=${CUDA_VERSION_%%.*} && \
    if [ "${CUDNN_VERSION_M}" == "9" ]; then \
        if [ "${SKIP_CUDNN}" = false ]; then \
            CUDNN_PACKAGE_NAME="${CUDNN_VERSION_}-1" && \
            apt-get install --only-upgrade -y \
            libcudnn9-headers-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
            libcudnn9-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
            libcudnn9-dev-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
            && \
            rm -rf /root/.cache && \
            rm -rf /var/lib/apt/lists/*; \
        else \
            apt-get purge -y libcudnn9-cuda-${CUDA_VERSION_M}; \
        fi \
    else \
        CUDNN_PACKAGE_NAME="${CUDNN_VERSION_}-1+cuda${CUDA_VERSION_%.*}" && \
        apt update -qq --fix-missing && \
        apt upgrade -y --allow-downgrades --allow-change-held-packages \
        libcudnn8=${CUDNN_PACKAGE_NAME} \
        libcudnn8-dev=${CUDNN_PACKAGE_NAME} \
        && \
        rm -rf /root/.cache && \
        rm -rf /var/lib/apt/lists/*; \
    fi

RUN sed -i '/nvidia-cudnn-frontend\|ninja/d' /etc/pip/constraint.txt
COPY pyproject.toml /tmp/
RUN pip install -U --no-cache-dir --group /tmp/pyproject.toml:dev
RUN pip install -U --no-cache-dir cupy-cuda13x

RUN pip install -U --no-cache-dir nvidia-cutlass-dsl apache-tvm-ffi torch-c-dlpack-ext cuda-python

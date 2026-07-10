
# Example usage
# export CUDNN_PATH=/home/scratch.agopal_sw/cudnn_frontend_0.9/debug_cudnn 
# export CUDA_PATH=/home/scratch.agopal_sw/cuda_12.8
# Usage: source /home/scratch.agopal_sw/lightning/cudnn_frontend/ci/manual/run_local_regression.sh fe_regress  --build_fe=true
# Usage: source /home/scratch.agopal_sw/lightning/cudnn_frontend/ci/manual/run_local_regression.sh fe_regress  --build_fe=false

# Function to display help message
show_help() {
    echo "Usage: "
    echo ""
    echo "export CUDNN_PATH=/path/to/cudnn  eg. export CUDNN_PATH=/home/scratch.agopal_sw/cudnn_frontend_0.9/debug_cudnn"
    echo "export CUDA_PATH=/path/to/cuda  eg. export CUDA_PATH=/home/scratch.agopal_sw/cuda_12.8"
    echo ""
    echo "source ci/manual/run_local_regression.sh [--help] --build_fe=true|false (default=false)"
    echo ""
    echo ""
    echo ""
    echo "This script takes one argument:"
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --build_fe=true|false (default=false)"
    echo ""
    echo "Example:"
    echo "source ci/manual/run_local_regression.sh fe_regress --build_fe=true"
    echo "To extract TK from your container use: https://confluence.nvidia.com/display/GCA/3.+Basic%3A+Build+cuDNN+with+Container#id-3.Basic:BuildcuDNNwithContainer-Testingoutsideofacontainer"
}

# Check for help argument
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    return 0
fi

# Check if correct number of arguments provided
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Expected 1 arguments, got $#"
    echo ""
    show_help
    return 0
fi

# Assign arguments to variables
ENV_NAME=$1

# Verify cudnn path exists
if [ ! -f "${CUDNN_PATH}/lib64/libcudnn.so" ]; then
    echo "Error: cudnn.so not found at ${CUDNN_PATH}/lib64/libcudnn.so"
    return 0
fi

# Verify cuda toolkit path exists
if [ ! -f "${CUDA_PATH}/lib64/libnvrtc.so" ]; then
    echo "Error: nvrtc.so not found at ${CUDA_TOOLKIT_PATH}/lib64/libnvrtc.so"
    return 0
fi

# Your script logic here
echo "Environment name: $ENV_NAME"
echo "CUDNN library path: $CUDNN_PATH"
echo "Cuda toolkit path: $CUDA_PATH"


# Create environment if it doesn't exist
if ! conda env list | grep -q "fe_regress"; then
    # Create environment if it doesn't exist
    conda create -n fe_regress python=3.10 -y
    echo "Environment fe_regress created with Python 3.10"
    conda init bash
else
    echo "Environment fe_regress already exists"
fi
eval "$(conda shell.bash hook)"
conda activate fe_regress

pip install --upgrade pip
pip uninstall -y pytest
pip install -r requirements.txt


if ! pip show torch > /dev/null 2>&1; then
    echo "PyTorch not found, installing..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    pip uninstall -y nvidia-cudnn-cu12
else
    echo "PyTorch is already installed"
fi

if [[ "$2" == *"true"* ]]; then
    echo "Building frontend"
    pip install -v .
else
    echo "Pulling pre-built frontend"
    pip install nvidia-cudnn-frontend
fi

export LD_LIBRARY_PATH=$CUDNN_PATH/lib64:$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo ""
echo "Now run pytest -s test/python/test_mhas_v2.py --unlock"
echo ""
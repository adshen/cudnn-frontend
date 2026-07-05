# rebuild docker images and tag them
# note that while CUDA and cuDNN versions are fixed,
# apt packages and pypi packages will use the latest versions
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.5.0.96_11.7.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=11.7.1 --build-arg CUDNN_VERSION_=8.5.0.96 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.6.0.163_11.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=11.8.0 --build-arg CUDNN_VERSION_=8.6.0.163 --build-arg DLFW_MONTH=22.10 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.7.0.84_11.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=11.8.0 --build-arg CUDNN_VERSION_=8.7.0.84 --build-arg DLFW_MONTH=23.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.8.1.3_12.0.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.0.1 --build-arg CUDNN_VERSION_=8.8.1.3 --build-arg DLFW_MONTH=23.03 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.2.26_12.1.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.1.0 --build-arg CUDNN_VERSION_=8.9.2.26 --build-arg DLFW_MONTH=23.05 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.5.29_12.2.2 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.2.2 --build-arg CUDNN_VERSION_=8.9.5.29 --build-arg DLFW_MONTH=24.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.7.29_12.2.2 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.2.2 --build-arg CUDNN_VERSION_=8.9.7.29 --build-arg DLFW_MONTH=24.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.0.0.312_12.3.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.3.1 --build-arg CUDNN_VERSION_=9.0.0.312 --build-arg DLFW_MONTH=24.02 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.1.0.70_12.4.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.4.0 --build-arg CUDNN_VERSION_=9.1.0.70 --build-arg DLFW_MONTH=24.03 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.2.0.82_12.4.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.4.0 --build-arg CUDNN_VERSION_=9.2.0.82 --build-arg DLFW_MONTH=24.05 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.2.1.18_12.5.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.5.0 --build-arg CUDNN_VERSION_=9.2.1.18 --build-arg DLFW_MONTH=24.06 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.3.0.75_12.5.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.5.1 --build-arg CUDNN_VERSION_=9.3.0.75 --build-arg DLFW_MONTH=24.07 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.4.0.58_12.6.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.6.0 --build-arg CUDNN_VERSION_=9.4.0.58 --build-arg DLFW_MONTH=24.08 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.5.1.17_12.6.2 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.6.2 --build-arg CUDNN_VERSION_=9.5.1.17 --build-arg DLFW_MONTH=24.10 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.6.0.74_12.6.3 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.6.3 --build-arg CUDNN_VERSION_=9.6.0.74 --build-arg DLFW_MONTH=24.11 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.7.0.66_12.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.8.0 --build-arg CUDNN_VERSION_=9.7.0.66 --build-arg DLFW_MONTH=25.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.8.0.87_12.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.8.0 --build-arg CUDNN_VERSION_=9.8.0.87 --build-arg DLFW_MONTH=25.02 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.9.0.52_12.9.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.9.0 --build-arg CUDNN_VERSION_=9.9.0.52 --build-arg DLFW_MONTH=25.04 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.10.1.4_12.9.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.9.0 --build-arg CUDNN_VERSION_=9.10.1.4 --build-arg DLFW_MONTH=25.04 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.11.0.98_12.9.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.9.1 --build-arg CUDNN_VERSION_=9.11.0.98 --build-arg DLFW_MONTH=25.06 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.12.0.46_12.9.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.9.1 --build-arg CUDNN_VERSION_=9.12.0.46 --build-arg DLFW_MONTH=25.06 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.13.0.50_13.0.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.0.0 --build-arg CUDNN_VERSION_=9.13.0.50 --build-arg DLFW_MONTH=25.08 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.14.0.64_13.0.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.0.0 --build-arg CUDNN_VERSION_=9.14.0.64 --build-arg DLFW_MONTH=25.08 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.15.0.58_13.0.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.0.0 --build-arg CUDNN_VERSION_=9.15.0.58 --build-arg DLFW_MONTH=25.08 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.16.0.29_13.0.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.0.0 --build-arg CUDNN_VERSION_=9.16.0.29 --build-arg DLFW_MONTH=25.10 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.17.1.4_13.1.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.1.0 --build-arg CUDNN_VERSION_=9.17.1.4 --build-arg DLFW_MONTH=25.12 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.18.1.3_13.1.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.1.0 --build-arg CUDNN_VERSION_=9.18.1.3 --build-arg DLFW_MONTH=25.12 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.19.0.56_13.1.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.1.0 --build-arg CUDNN_VERSION_=9.19.0.56 --build-arg DLFW_MONTH=26.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.20.0.48_13.2.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.2 --build-arg CUDNN_VERSION_=9.20.0.48 --build-arg DLFW_MONTH=26.02 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.21.0.82_13.2.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.2 --build-arg CUDNN_VERSION_=9.21.0.82 --build-arg DLFW_MONTH=26.03 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.22.0.52_13.2.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.2 --build-arg CUDNN_VERSION_=9.22.0.52 --build-arg DLFW_MONTH=26.04 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.23.0.39_13.3.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.3 --build-arg CUDNN_VERSION_=9.23.0.39 --build-arg DLFW_MONTH=26.05 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.24.0.43_13.3.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.3 --build-arg CUDNN_VERSION_=9.24.0.43 --build-arg DLFW_MONTH=26.06 .

docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_12.9.1 -f dockers/Dockerfile --build-arg CUDA_VERSION_=12.9.1 --build-arg CUDNN_VERSION_=9.12.0.46 --build-arg SKIP_CUDNN=true --build-arg DLFW_MONTH=25.06 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_13.3.0 -f dockers/Dockerfile --build-arg CUDA_VERSION_=13.3 --build-arg CUDNN_VERSION_=9.23.0.39 --build-arg SKIP_CUDNN=true --build-arg DLFW_MONTH=26.05 .

#############################################
############# RUN WITH CAUTION ##############
#############################################
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.5.0.96_11.7.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.6.0.163_11.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.7.0.84_11.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.8.1.3_12.0.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.2.26_12.1.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.5.29_12.2.2
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.7.29_12.2.2
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.0.0.312_12.3.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.1.0.70_12.4.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.2.0.82_12.4.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.2.1.18_12.5.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.3.0.75_12.5.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.4.0.58_12.6.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.5.1.17_12.6.2
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.6.0.74_12.6.3
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.7.0.66_12.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.8.0.87_12.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.9.0.52_12.9.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.10.1.4_12.9.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.11.0.98_12.9.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.12.0.46_12.9.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.13.0.50_13.0.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.14.0.64_13.0.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.15.0.58_13.0.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.16.0.29_13.0.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.17.1.4_13.1.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.18.1.3_13.1.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.19.0.56_13.1.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.20.0.48_13.2.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.21.0.82_13.2.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.22.0.52_13.2.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.24.0.43_13.3.0

docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_12.9.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_13.3.0

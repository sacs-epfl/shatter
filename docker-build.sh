#!/bin/bash

# Specify the target platform
#TARGET_PLATFORM="linux/amd64"

# Build and push the image using Docker Buildx
docker buildx build -f Dockerfile --build-arg TORCH_CUDA_ARCH_LIST="Ampere;Turing;Volta" \
    -t shatter-artifacts \
    --load .
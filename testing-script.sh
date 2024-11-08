#!/bin/bash

set -euo pipefail

#Check CUDA is available, if CUDA is installed and nvcc is not available, install nvidia-cuda-toolkit.
nvcc --version
echo "For smoother execution, the output above should match Cuda compilation tools, release 12.3, V12.3.107\nBuild cuda_12.3.r12.3/compiler.33567101_0"

#Check that transformers and torch can be imported in python venv
echo "Activating Conda"
source ${CONDA_PREFIX}/bin/activate
conda activate venv && conda list | grep transformers
conda activate venv && conda list | grep torch

# Check the installation of libgl1-mesa-glx
if ! dpkg-query -W -f='${Status}' libgl1-mesa-glx 2>/dev/null | grep -q "install ok installed"; then
    echo "Error: libgl1-mesa-glx is not installed. Install it with sudo apt-get install libgl1-mesa-glx. This is needed for gradient inversion experiments." >&2
else
    echo "libgl1-mesa-glx is installed."
fi

result=True
echo "Checking models exist" 
for dir in \
        "$PWD/artifact_scripts/gradientInversion/rog/data/val" \
        "$PWD/artifact_scripts/gradientInversion/rog/model_zoos" 
do
    if ! [ -d "$dir" ]; then
        result=False
        echo "${dir} model does not exist: check that the data.zip file is unzipped properly" 
        break
    fi
done

result=True
echo "Checking three datasets exist" 
for dir in \
        "$PWD/eval/data/CIFAR10" \
        "$PWD/eval/data/movielens" \
        "$PWD/eval/data/sent140" 
do
    if ! [ -d "$dir" ]; then
        result=False
        echo "${dir} dataset does not exist: check that git-lfs is installed and pull the dataset from repo using it." 
        break
    fi
done


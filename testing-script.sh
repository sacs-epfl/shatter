#!/bin/bash

set -euo pipefail

#Check CUDA is available, if CUDA is installed and nvcc is not available, install nvidia-cuda-toolkit.
nvcc --version

#Check that transformers and torch can be imported in python venv 
conda activate venv && conda list | grep transformers
conda activate venv && conda list | grep torch

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


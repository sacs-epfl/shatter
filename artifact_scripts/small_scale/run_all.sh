#!/bin/bash

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root of shatter repository> <environment python executable folder, e.g. ~/miniconda/envs/shatter/bin/>"
    exit 1
fi

nfs_home=$1
python_bin=$2

./run_CIFAR10.sh $nfs_home $python_bin

./run_Movielens.sh $nfs_home $python_bin

./run_Twitter.sh $nfs_home $python_bin

echo Done Everything!

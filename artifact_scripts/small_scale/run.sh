#!/bin/bash

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root of shatter repository> <environment python executable folder, e.g. ~/miniconda/envs/shatter/bin/>"
    exit 1
fi

nfs_home=$1
python_bin=$2

echo Computing CIFAR10/EL
CIFAR10/EL/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py CIFAR10/EL

echo Computing CIFAR10/Muffliato05
CIFAR10/Muffliato05/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py CIFAR10/Muffliato05

echo Computing CIFAR10/VNodes2
CIFAR10/VNodes2/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py CIFAR10/VNodes2

echo Computing Movielens/EL
Movielens/EL/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Movielens/EL

echo Computing Movielens/Muffliato025
Movielens/Muffliato025/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Movielens/Muffliato025

echo Computing Movielens/VNodes2
Movielens/VNodes2/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Movielens/VNodes2

echo Computing Twitter/EL
Twitter/EL/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Twitter/EL

echo Computing Twitter/Muffliato009
Twitter/Muffliato009/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Twitter/Muffliato009

echo Computing Twitter/VNodes2
Twitter/VNodes2/run.sh $nfs_home $python_bin
eval/plot_csv_acc.py Twitter/VNodes2

echo Done Everything!

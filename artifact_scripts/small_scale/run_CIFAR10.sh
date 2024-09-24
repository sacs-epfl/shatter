#!/bin/bash

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <absolute path of root of shatter repository> <environment python executable folder, e.g. ~/.conda/envs/venv/bin/>"
    exit 1
fi

nfs_home=$1
python_bin=$2

echo Computing CIFAR10/EL
CIFAR10/EL/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py CIFAR10/EL
$python_bin/python $nfs_home/eval/evaluate_attack.py CIFAR10/EL

echo Computing CIFAR10/Muffliato05
CIFAR10/Muffliato05/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py CIFAR10/Muffliato05
$python_bin/python $nfs_home/eval/evaluate_attack.py CIFAR10/Muffliato05

echo Computing CIFAR10/VNodes2
CIFAR10/VNodes2/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py CIFAR10/VNodes2
$python_bin/python $nfs_home/eval/evaluate_attack.py CIFAR10/VNodes2

echo Done CIFAR10!

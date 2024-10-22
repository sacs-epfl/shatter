#!/bin/bash

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root of shatter repository> <environment python executable folder, e.g. ~/.conda/envs/venv/bin/>"
    exit 1
fi

nfs_home=$1
python_bin=$2


echo Computing Movielens/EL
Movielens/EL/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Movielens/EL
$python_bin/python $nfs_home/eval/evaluate_attack.py Movielens/EL

echo Computing Movielens/Muffliato025
Movielens/Muffliato025/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Movielens/Muffliato025
$python_bin/python $nfs_home/eval/evaluate_attack.py Movielens/Muffliato025

echo Computing Movielens/VNodes2
Movielens/VNodes2/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Movielens/VNodes2
$python_bin/python $nfs_home/eval/evaluate_attack.py Movielens/VNodes2

echo Done Movielens!

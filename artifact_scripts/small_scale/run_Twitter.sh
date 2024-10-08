#!/bin/bash

# Check if the number of arguments is exactly 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <root of shatter repository> <environment python executable folder, e.g. ~/.conda/envs/venv/bin/>"
    exit 1
fi

nfs_home=$1
python_bin=$2

echo Computing Twitter/EL
Twitter/EL/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Twitter/EL
$python_bin/python $nfs_home/eval/evaluate_attack.py Twitter/EL

echo Computing Twitter/Muffliato009
Twitter/Muffliato009/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Twitter/Muffliato009
$python_bin/python $nfs_home/eval/evaluate_attack.py Twitter/Muffliato009

echo Computing Twitter/VNodes2
Twitter/VNodes2/run.sh $nfs_home $python_bin
$python_bin/python $nfs_home/eval/plot_csv_acc.py Twitter/VNodes2
$python_bin/python $nfs_home/eval/evaluate_attack.py Twitter/VNodes2

echo Done Twitter!

#!/bin/bash

# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment python executable folder, e.g. ~/miniconda/envs/shatter/bin/>"
    exit 1
fi

env_python=$1/python

# Compute the results

config_file=config_fedavg_lenet.yaml
$env_python attack_fedavg.py $config_file

config_file=config_topk_lenet.yaml
$env_python attack_fedavg.py $config_file

config_file=config_vnodes2_lenet.yaml
$env_python attack_vnodes.py $config_file 2

config_file=config_vnodes4_lenet.yaml
$env_python attack_vnodes.py $config_file 4

config_file=config_vnodes8_lenet.yaml
$env_python attack_vnodes.py $config_file 8

config_file=config_vnodes16_lenet.yaml
$env_python attack_vnodes.py $config_file 16


echo "Reconstructed images can be found in experiments/"

echo "Aggregating Metrics"

$env_python aggregateLPIPS.py

echo "LPIPS plot and CSV generated!"

#!/bin/bash

# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <environment python executable folder, e.g. ~/.conda/envs/venv/bin/>"
    exit 1
fi

env_python=$1/python
num_clients=1

# Compute the results

config_file=config_fedavg_lenet.yaml
$env_python attack_fedavg.py $config_file $num_clients

echo "Finished EL Reconstructions!"
sleep 2

config_file=config_topk_lenet.yaml
$env_python attack_fedavg.py $config_file $num_clients
echo "Finished TopK Reconstructions!"
sleep 2

for num_chunks in 2 4 8 16
do
    config_file=config_vnodes"${num_chunks}"_lenet.yaml
    $env_python attack_vnodes.py $config_file $num_clients $num_chunks
    echo "Finished Shatter with k=$num_chunks Reconstructions!"
    sleep 2
done


echo "Reconstructed images can be found in experiments/"

echo "Aggregating Metrics"

$env_python aggregateLPIPS.py

echo "LPIPS plot and CSV generated!"

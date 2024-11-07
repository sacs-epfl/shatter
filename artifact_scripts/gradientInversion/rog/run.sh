#!/bin/bash

num_clients=1

# Compute the results

config_file=config_fedavg_lenet.yaml
python attack_fedavg.py $config_file $num_clients

echo "Finished EL Reconstructions!"
sleep 2

config_file=config_topk_lenet.yaml
python attack_fedavg.py $config_file $num_clients
echo "Finished TopK Reconstructions!"
sleep 2

for num_chunks in 2 4 8 16
do
    config_file=config_vnodes"${num_chunks}"_lenet.yaml
    python attack_vnodes.py $config_file $num_clients $num_chunks
    echo "Finished Shatter with k=$num_chunks Reconstructions!"
    sleep 2
done


echo "Reconstructed images can be found in experiments/"

echo "Aggregating Metrics"

python aggregateLPIPS.py

echo "LPIPS plot and CSV generated!"

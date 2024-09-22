#!/bin/bash

nfs_home=$1
python_bin=$2
logs_subfolder="artifact_scripts/small_scale/Movielens/VNodes2"
decpy_path=$nfs_home/eval
mkdir -p ~/tmp
cd $decpy_path

env_python=$python_bin/python
procs_per_machine=8
config_file=~/tmp/config.ini
machines=1
iterations=501
eval_file=testingSimulation.py
log_level=INFO

ip_machines=$nfs_home/$logs_subfolder/ip.json

m="0"

echo -e $procs_per_machine

# Base configs for which the gird search is done
tests=("$nfs_home/$logs_subfolder/config_VNodes2.ini")


procs=`expr $procs_per_machine \* $machines`
echo procs: $procs
# Calculating the number of samples that each user/proc will have on average

# random_seeds for which to rerun the experiments
random_seeds=("90")

echo iterations: $iterations

test_after=100
train_evaluate_after=100

echo test after: $test_after

exit_code=0
for i in "${tests[@]}"
do
  for seed in "${random_seeds[@]}"
  do
    echo $i
    IFS='_' read -ra NAMES <<< $i
    IFS='.' read -ra NAME <<< ${NAMES[-1]}
    log_suffix=${NAME[0]}:r=$comm_rounds_per_global_epoch:b=$batchsize:$(date '+%Y-%m-%dT%H:%M')
    log_dir_base=$nfs_home/$logs_subfolder/$log_suffix
    echo results are stored in: $log_dir_base
    log_dir=$log_dir_base/machine$m
    mkdir -p $log_dir
    weight_store_dir=$log_dir_base/weights
    mkdir -p $weight_store_dir
    cp $i $config_file
    $python_bin/crudini --set $config_file COMMUNICATION addresses_filepath $ip_machines
    $python_bin/crudini --set $config_file DATASET random_seed $seed
    $env_python $eval_file -ro 0 -tea $train_evaluate_after -ld $log_dir -wsd $weight_store_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $config_file -ll $log_level #> $log_dir/console.log 2>error.log
    ((exit_code=$exit_code|$?))
    echo $i is done
    done
done
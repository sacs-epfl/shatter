#!/bin/bash

set -euxo pipefail

# Check if the number of arguments is exactly 8
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <number of processes> <number of iterations e.g. 51> <full path of config file (.ini)> <full path of testing_XXX.py in the eval folder> <No. of comm. rounds between each test eval> <No. of comm. rounds between each train eval> <Train_dataset_dir> <Test_dataset_dir>"
    exit 1
fi

procs_per_machine=$1
iterations=$2
tests=$3
eval_file=$4
test_after=$5
train_evaluate_after=$6
train_dataset_dir=$7
test_dataset_dir=$8

# nfs_home=$SHATTER_HOME
# decpy_path=$nfs_home/eval
mkdir -p ~/tmp

config_file=~/tmp/config.ini
machines=1
log_level=INFO

ip_machines=$(pwd)/ip.json

m="0"

echo -e $procs_per_machine

procs=`expr $procs_per_machine \* $machines`
echo procs: $procs

# random_seeds for which to rerun the experiments
random_seeds=("90")

echo iterations: $iterations


exit_code=0
for i in "${tests[@]}"
do
  for seed in "${random_seeds[@]}"
  do
    IFS='_' read -ra NAMES <<< $i
    IFS='.' read -ra NAME <<< ${NAMES[-1]}
    log_basename=$(echo "${i##*/config_}" | sed 's/\.ini$//')
    log_suffix="$log_basename"-$(date '+%Y-%m-%dT%H:%M')
    log_dir_base=$(pwd)/$log_suffix
    echo results are stored in: $log_dir_base
    log_dir=$log_dir_base/machine$m
    mkdir -p $log_dir
    weight_store_dir=$log_dir_base/weights
    mkdir -p $weight_store_dir
    cp $i $config_file
    crudini --set $config_file COMMUNICATION addresses_filepath $ip_machines
    crudini --set $config_file DATASET random_seed $seed
    crudini --set $config_file DATASET train_dir $train_dataset_dir
    crudini --set $config_file DATASET test_dir $test_dataset_dir
    # cd $decpy_path
    python $eval_file -ro 0 -tea $train_evaluate_after -ld $log_dir -wsd $weight_store_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -ta $test_after -cf $config_file -ll $log_level #> $log_dir/console.log 2>error.log
    echo $i is done
    done
done
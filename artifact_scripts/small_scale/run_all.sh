#!/bin/bash

set -euxo pipefail

cd $SHATTER_HOME/artifact_scripts/small_scale

./run_CIFAR10.sh

./run_Movielens.sh

./run_Twitter.sh
echo "Done Everything!"

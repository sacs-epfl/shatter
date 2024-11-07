#!/bin/bash

set -euxo pipefail

./run_CIFAR10.sh

./run_Movielens.sh

./run_Twitter.sh
echo "Done Everything!"

#!/bin/bash

set -euxo pipefail

experimentPath=artifact_scripts/small_scale/

mkdir -p ./$experimentPath/results

docker cp shatter-artifacts:/root/shatter/$experimentPath/{CIFAR10,Movielens,Twitter} ./$experimentPath/results

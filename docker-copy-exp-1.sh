#!/bin/bash

set -euxo pipefail

experimentPath=artifact_scripts/gradientInversion/rog/experiments

mkdir -p ./$experimentPath

docker cp shatter-artifacts:/root/shatter/$experimentPath ./$experimentPath

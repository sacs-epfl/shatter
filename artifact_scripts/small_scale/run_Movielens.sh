#!/bin/bash

set -euxo pipefail

# Check if the 'conda' command is available
if ! command -v conda &> /dev/null; then
    echo "Activating Conda"
    source ${CONDA_PREFIX}/bin/activate
fi

conda activate venv

echo Computing EL on Movielens
cd $SHATTER_HOME/artifact_scripts/small_scale/Movielens
$SHATTER_HOME/eval/run_helper.sh 8 501 $(pwd)/config_EL.ini $SHATTER_HOME/eval/testingSimulation.py 100 100 $SHATTER_HOME/eval/data/movielens $SHATTER_HOME/eval/data/movielens

echo Computing Muffliato on Movielens
$SHATTER_HOME/eval/run_helper.sh 8 5001 $(pwd)/config_Muffliato025.ini $SHATTER_HOME/eval/testingSimulation.py 100 100 $SHATTER_HOME/eval/data/movielens $SHATTER_HOME/eval/data/movielens

echo Computing Shatter on Movielens
$SHATTER_HOME/eval/run_helper.sh 8 501 $(pwd)/config_Shatter2.ini $SHATTER_HOME/eval/testingSimulation.py 100 100 $SHATTER_HOME/eval/data/movielens $SHATTER_HOME/eval/data/movielens

echo "Making CSVs and Plots for Movielens in ./Movielens"
python $SHATTER_HOME/eval/plot_csv_acc.py .
python $SHATTER_HOME/eval/evaluate_attack.py .

echo Done Movielens!

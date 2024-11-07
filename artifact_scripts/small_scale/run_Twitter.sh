#!/bin/bash

set -euxo pipefail

echo Computing EL on Twitter
cd $SHATTER_HOME/artifact_scripts/small_scale/Twitter
$SHATTER_HOME/eval/run_helper.sh 4 51 $(pwd)/config_EL.ini $SHATTER_HOME/eval/testingSimulation.py 10 10 $SHATTER_HOME/eval/data/sent140/train $SHATTER_HOME/eval/data/sent140/test

echo Computing Muffliato on Twitter
$SHATTER_HOME/eval/run_helper.sh 4 501 $(pwd)/config_Muffliato009.ini $SHATTER_HOME/eval/testingMuffliato.py 10 10 $SHATTER_HOME/eval/data/sent140/train $SHATTER_HOME/eval/data/sent140/test

echo Computing Shatter on Twitter
$SHATTER_HOME/eval/run_helper.sh 4 51 $(pwd)/config_Shatter2.ini $SHATTER_HOME/eval/testingSimulation.py 10 10 $SHATTER_HOME/eval/data/sent140/train $SHATTER_HOME/eval/data/sent140/test

echo Making CSVs and Plots for Twitter in ./Twitter
python $SHATTER_HOME/eval/plot_csv_acc.py .
python $SHATTER_HOME/eval/evaluate_attack.py .

echo Done Twitter!

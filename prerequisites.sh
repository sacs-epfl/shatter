#!/bin/bash

set -euxo pipefail

if [ -z "$SHATTER_HOME" ]
then
    echo "The environment variable SHATTER_HOME is not defined. Set it to the root of the shatter repository."
else
    echo "The environment variable SHATTER_HOME is set to: $SHATTER_HOME"
fi

if [ -z "$CONDA_PREFIX" ]
then
    # Install Miniconda
    export CONDA_PREFIX=$SHATTER_HOME/.conda
    echo "Installing Miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ${CONDA_PREFIX}
    rm miniconda.sh
    # mkdir -p $SHATTER_HOME/.conda && 
else
    echo "The environment variable CONDA_PREFIX is set to: $CONDA_PREFIX"
fi

echo "Activating Conda"
source ${CONDA_PREFIX}/bin/activate
conda init --all

# Set the working directory to the root of the repo
cd $SHATTER_HOME

echo "Creating a Python virtual environment"
# Setting up and activating a Python virtual environment with Python 3.10
conda create -n venv python=3.10
conda activate venv

echo "Installing Python dependencies through Conda"
# Upgrading pip and setuptools
conda run -n venv pip install --upgrade pip
conda run -n venv pip install --upgrade setuptools
conda run -n venv pip install -r requirements.txt

cd $SHATTER_HOME/artifact_scripts/gradientInversion/rog/
echo "If the following command fails to build wheels, libgl1-mesa-glx is missing. Just apt install it."
conda run -n venv pip install -r requirements.txt
unzip data.zip
rm data.zip

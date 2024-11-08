#!/bin/bash

set -euxo pipefail

if [ -z "$SHATTER_HOME" ]
then
    echo "The environment variable SHATTER_HOME is not defined. Set it to the root of the shatter repository."
else
    echo "The environment variable SHATTER_HOME is set to: $SHATTER_HOME"
fi

if [ -d "${CONDA_PREFIX}" ]
then
    echo "The environment variable CONDA_PREFIX is set to: $CONDA_PREFIX"
else
    # Install Miniconda
    echo "Installing Miniconda"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p ${CONDA_PREFIX}
    rm miniconda.sh
fi

echo "Activating Conda"
source ${CONDA_PREFIX}/bin/activate
conda init --all

# Set the working directory to the root of the repo
cd $SHATTER_HOME

echo "Creating a Python virtual environment"
# Setting up and activating a Python virtual environment with Python 3.11
conda create -y -n venv python=3.11
conda activate venv

echo "Installing Python dependencies through Conda"
# Upgrading pip and setuptools
conda run -n venv pip install --upgrade pip
conda run -n venv pip install --upgrade setuptools
conda run -n venv pip install -r requirements_base.txt
conda run -n venv pip install -r requirements_all.txt


# Unzip ImageNet data for the gradient inversion
cd $SHATTER_HOME/artifact_scripts/gradientInversion/rog/
#echo "If the following command fails to build wheels, libgl1-mesa-glx is missing. Just apt install it."
unzip data.zip
rm data.zip
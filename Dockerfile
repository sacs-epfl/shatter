# Use the NVIDIA CUDA image for the specified platform
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Update and install required packages
RUN apt-get update && apt-get install -y \
    openssl \
    bash \
    curl \
    vim \
    sudo \
    iproute2 \
    wget \
    git \
    tmux \
    htop \
    psmisc \
    openssh-server \
    nodejs \
    zip \
    unzip \
    libgl1-mesa-glx \
    git-lfs

# Set the home directory
ENV HOME=/root
# ENV TORCH_CUDA_ARCH_LIST=Turing
# Create a directory called 'shatter' in the home directory
RUN mkdir -p $HOME/shatter

# Copy the contents of the current folder into the 'shatter' directory
COPY . $HOME/shatter

# Unzip ImangeNet Validation Data
WORKDIR $HOME/shatter/artifact_scripts/gradientInversion/rog/
RUN unzip data.zip
RUN rm data.zip

WORKDIR $HOME
# Install Miniconda
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}

RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN rm miniconda.sh

# Set the working directory to the 'shatter' directory
WORKDIR $HOME/shatter

# Install the required packages
RUN ${CONDA} create -n venv python=3.10

ENV VENV_PATH=${CONDA_PREFIX}/envs/venv/bin

RUN ${CONDA} run -n venv pip install --upgrade pip
RUN ${CONDA} run -n venv pip install --upgrade setuptools
RUN ${CONDA} run -n venv pip install -r requirements.txt
# WORKDIR $HOME/shatter/artifact_scripts/gradientInversion/rog/
# RUN ${CONDA} run -n venv pip install -r requirements.txt

WORKDIR $HOME/shatter

# Default command to open a bash shell
CMD ["/bin/bash"]
# Use the NVIDIA CUDA image for the specified platform
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Pytorch requires the CUDA SDK architecture version to be specified
#  for Docker-based Pytorch builds to work
# Set this argument as:
#  docker build --build-arg TORCH_CUDA_ARCH_LIST=Turing .
ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}

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

# Create a directory called 'shatter' in the home directory
RUN mkdir -p /root/shatter

# Set the home directory
ENV SHATTER_HOME=/root/shatter

# Copy the contents of the current folder into the 'shatter' directory
COPY . $SHATTER_HOME

WORKDIR $SHATTER_HOME
# This is required to install Conda
ENV CONDA_PREFIX=${SHATTER_HOME}/.conda

RUN bash prerequisites.sh

RUN bash testing-script.sh

# Run experiment 1
#RUN cd $SHATTER_HOME/artifact_scripts/gradientInversion/rog && ./run.sh
## Run experiment 2
#RUN cd $SHATTER_HOME/artifact_scripts/small_scale && ./run_all

# Default command to open a bash shell
CMD ["/bin/bash"]
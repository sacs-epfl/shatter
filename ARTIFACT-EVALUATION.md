# Artifact Appendix

Paper title: **Noiseless Privacy-Preserving Decentralized Learning**

Artifacts HotCRP Id: **#11**

Requested Badge: **Functional**

## Description
The artifact provides scripts and code supporting the results of the paper. Since each experiment was executed at a large scale and for a long time, we provide scripts to execute a small-scale version of the experiments.

### Security/Privacy Issues and Ethical Concerns (All badges)
No issues.

## Basic Requirements (Only for Functional and Reproduced badges)
To run the small scale version of the artifact, a machine with a GPU and DRAM of 64 GB should suffice.
Out of memory issues result in processes waiting for messages from killed processes and is hard to spot.
To run the experiments at the full scale, one requires access to 25 g5.2x large instances from AWS. We unfortunately cannot provide access to these.

### Hardware Requirements
An Nvidia GPU is required to run this artifact.
The artifacts were tested with Nvidia A100 80GB GPU, NVIDIA T4 GPUs and NVIDIA Corporation GA100GL A30 PCIe (rev a1).
Readers can determine that they have a CUDA compatible GPU using the following command, based on the [Nvidia Cuda installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-you-have-a-cuda-capable-gpu).
You might want to install `pciutils` if command `lspci` is not found.

```shell
lspci | grep -i nvidia
```


### Software Requirements
We tested the artifacts on Ubuntu 22.04, Python 3.11, and CUDA 12.3.
The project can be built on the host OS of the machine or within a Docker container (For docker installation, please refer to the [official documentation](https://www.docker.com/get-started/)).
For both methods, the [Nvidia CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) is needed to build and execute applications using the Nvidia GPUs. This includes the CUDA runtime, CUDA driver and the CUDA C++ API. Nvidia CUDA toolkit can be downloaded and installed using the link above. Check that the CUDA driver and CUDA runtimes are installed correctly, and fetch their versions. Refer to [the post-installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#running-the-binaries).

Once you have verified that you have an Nvidia GPU, check the CUDA version with:

```shell
nvcc --version
```

The output should look like the following:


```shell
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

Consult the [Nvidia compatibility support page FAQs](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#frequently-asked-questions) to determine whether the project can be built when the host OS has different versions of CUDA drivers. 

We recommend to update the CUDA driver to 12.3 with the following steps in [this blog](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202):

```shell
sudo apt update && sudo apt upgrade
sudo apt autoremove nvidia* --purge
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-driver-565
sudo reboot

sudo apt update && sudo apt upgrade
sudo apt install nvidia-cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12

```

#### Requirements for Building with Docker
The project can also be built with Docker.
For this, please first install Docker by followiung the official website: [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/).

[A Beginner’s Guide to NVIDIA Container Toolkit on Docker](https://medium.com/@u.mele.coding/a-beginners-guide-to-nvidia-container-toolkit-on-docker-92b645f92006) is a good reference to getting started with CUDA Docker containers. We describe important steps below.

In addition to the CUDA toolkit installed above, install the [Nvidia Container toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to pass through the GPU drivers to the container engine (Docker daemon). Please refer to the [official website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) to download this toolkit and configure Docker to use it. 
Remember to restart the Docker daemon after installing the toolkit.

Adding Nvidia GPG Keys and Repository:
```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg   
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | 
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |  
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Installing the toolkit and restarting docker:
```shell
sudo apt-get update && \
sudo apt-get install -y nvidia-container-toolkit && \
sudo nvidia-ctk runtime configure --runtime=docker  && \
sudo systemctl restart docker
```

Test the correctness of the docker + cuda installation with the following docker container and check that the GPU is detected within the container:
```shell
docker run --rm --gpus all nvidia/cuda:12.3.2-devel-ubuntu22.04 nvcc --version && nvidia-smi
```
This should give you the CUDA version 12.3 and your GPU should show up in the `nvidia-smi` output.


### Estimated Time and Storage Consumption
Each experiment should take roughly an hour. So, in total, the experiments should take ~5.5 hours.
Each experiment should take up roughly 200MB of storage, totalling up to ~4GBs of storage including the datasets and the models (used for eval).
All time measurements were done when using 8 CPU cores and 1 Nvidia A100 80GB GPU.

## Environment

### Accessibility (All badges)
The artifact code and data can be accessed via [https://github.com/sacs-epfl/shatter](https://github.com/sacs-epfl/shatter). This is the lab's public Github.
The code is published under the [MIT License](https://github.com/sacs-epfl/shatter/blob/main/LICENSE).
We also provide a Dockerfile in the repository and the docker image on the Docker Hub as `rishis8/shatter-artifact-pets2025:latest`.
The provided Docker image uses a base Nvidia image, and hence, its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license: [https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).

When cloning directly from the Github repository, git-lfs is required to download the datasets and models.
[Readers can install Git-Lfs following these official instructions](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#getting-started).
Use `git lfs pull` to ensure large files are downloaded after cloning.

```shell
    git clone https://github.com/sacs-epfl/shatter.git && cd shatter && \
    git switch -c shatter-pets-2025 && \
    git lfs pull
```

### Set up the environment (Only for Functional and Reproduced badges)
We recommend using the Docker image (`rishis8/shatter-artifact-pets2025:latest`) since everything is already set up. In all further instructions, we assume that the default directory is the `shatter` repository root. In the docker image, this is `/root/shatter`.

You can also build your own docker image with the provided Dockerfile.
Importantly, _before_ building the Docker image, determine the version of the CUDA SDK (Refer to Software Requirements section above).
This is required for Pytorch to compile within Docker.  Consult [this matrix](https://en.wikipedia.org/wiki/CUDA#GPUs_supported), and given the CUDA SDK version, determine the CPU microarchitecture family, such as _Turing_ or _Volta_. 
When building the Docker image, use build arguments to set the ```TORCH_CUDA_ARCH_LIST``` environment variable for the build. (Official Pytorch docs on how this environment variable is used can be found [here](https://pytorch.org/docs/stable/cpp_extension.html).) The ```SHATTER_HOME``` and ```CONDA_PREFIX``` environment variables are set automatically within the Dockerfile.

In `docker-build.sh`, update the ```TOTCH_CUDA_ARCH_LIST``` with your microarchitecture, and then, run:
```shell
./docker-build.sh
```

After the docker build completes, remember to check your installation of Nvidia container toolkit as described in [Software Requirements](#software-requirements) above. The `nvidia-smi` and `nvcc --version` commands should succeed from within the container (See [Section Requirements for Building with Docker](#requirements-for-building-with-docker) above).

To run the image, use the following command:
```shell
./docker-run.sh
```
To run the prebuilt image, replace the target (flagged with -t) in `docker-run.sh` from ```shatter-artifacts``` to ```rishis8/shatter-artifact-pets2025:latest```.

#### Setup without Docker
It is important to install ```libgl1-mesa-glx```.

```shell
sudo apt-get update && sudo apt-get -y install libgl1-mesa-glx
```

If not using docker, set ```$SHATTER_HOME``` to the root of `shatter` repository.
Set ```CONDA_PREFIX``` if not already set to the desired installation location for [conda](https://docs.anaconda.com/miniconda/).

Then set up the environment with the available script:

```shell
./prerequisites.sh
```

### Testing the Environment (Only for Functional and Reproduced badges)
When using Docker, check the Host and Container are working correctly with GPUs as described in [Section Requirements for Building with Docker](#requirements-for-building-with-docker) of this file.

Finally, use the `testing-script.sh` to see if everything is correct:
```shell
./testing-script.sh
```
The output should look like this:
```
(base) root@69d0f26da68f:~/shatter# ./testing-script.sh
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
For smoother execution, the output above should match Cuda compilation tools, release 12.3, V12.3.107\nBuild cuda_12.3.r12.3/compiler.33567101_0
Activating Conda
transformers              4.46.2                   pypi_0    pypi
torch                     2.5.1                    pypi_0    pypi
torchvision               0.20.1                   pypi_0    pypi
libgl1-mesa-glx is installed.
Checking models exist
Checking three datasets exist
```

## Artifact Evaluation (Only for Functional and Reproduced badges)
This section includes all the steps required to evaluate your artifact's functionality and validate your paper's key results and claims.
Therefore, highlight your paper's main results and claims in the first subsection. And describe the experiments that support your claims in the subsection after that.

### Main Results and Claims
Shatter achieves better accuracy and privacy compared to the baselines of EL and Muffliato.

#### Main Result 1: Gradient-inversion attacks
Shatter provides stronger defence against gradient inversion attacks.
We measure this in terms of LPIPS score (higher is more private) and visually by inspecting the images.
In the paper, this is shown in Figure 2 (Section 3) and Figure 8 (Section 6.4).

#### Main Result 2: Membership-inference and Linkability attacks
Shatter provides stronger defence against membership inference attack (MIA) as evident by the lower AUC of the ROC curve.
Furthermore, Shatter lowers the linkability attack (LA) success as evident by the lower attack accuracy with Shatter as compared to EL and Muffliato.
Sections 6.2 and 6.3 demonstrate this.

### Experiments 

#### Experiment 1: Gradient-inversion attack
For Experiment 1, run the following command:
```shell
$SHATTER_HOME/artifact_scripts/gradientInversion/rog/run.sh
```
This should take ~15 minutes and about 30 MBs of space because of reconstructed images.
Reconstructed images per client, aggregated data CSVs and bar plots are generated in `$SHATTER_HOME/artifact_scripts/gradientInversion/rog/experiments/lenet`.

Some additional details:
- VNodes{k} is Shatter with k virtual nodes.
- The reconstructed images and lpips scores can be compared to Figures 2 and 8. Furthermore, lpips_bar_plot.png is analogous to Figure 7(d). You can ignore other metrics like `snr` and `ssim`. LPIPS will not be exact numbers in the paper since only 1 client was attacked as opposed to 100 in the experiments in the paper.
- We recommend clearing up `artifact_scripts/gradientInversion/rog/experiments/lenet` before running other experiments to save disk space.
- If you get a `ModuleNotFoundError`, verify the conda environment `venv` is active and you followed the steps in the [Setting up the Environment Section](#set-up-the-environment-only-for-functional-and-reproduced-badges).

#### Experiment 2: Convergence, MIA and LA
These experiments are smaller scale versions of the other experiments in the paper since the full-scale experiments take very long and need to be run across 25 machines. To run experiment 2, execute the following command:
```shell
$SHATTER_HOME/artifact_scripts/small_scale/run_all.sh
```
This runs the experiments for all the datasets in one go.

To do this step by step, one can also individually run the scripts for each dataset in `$SHATTER_HOME/artifact_scripts/small_scale`.

Experiments with CIFAR-10 and Movielens datasets should take ~1.5 hour and ~200MBs in disk space each. Twitter dataset experiments take a bit longer and can take ~2.5 hours and ~200 MBs. In total `run_all.sh` should run in ~5.5 hours and ~600MBs of disk space.
Inside `$SHATTER_HOME/artifact_scripts/small_scale/CIFAR10`, the aggregated CSVs for each baseline can be found: `*test_acc.csv` (Figure 3, 5, 7 all except Movielens), `*test_loss.csv` (Figure 3, 5, 7 Movielens), `*clients_linkability.csv` (Figure 6), `*clients_MIA.csv` (Figure 6), `*iterations_linkability.csv` (Partially Figure 7c), and `*iterations_MIA.csv` (Figure 5). PDFs for the plots with all baselines together (not exactly the ones in the paper, but same figures as the CSVs) are also created in the same folders. Since these are smaller scale experiments, the values will not match the ones in the paper.

Things to watch out for:
- If CUDA OOM is encountered, try lowering the `test_batch_size` and `batch_size` in `config*.ini` within each dataset and baseline folder. One such `config` file is `$SHATTER_HOME/artifact_scripts/small_scale/CIFAR10/EL/config_EL.ini`
- If the experiments look like they are in a deadlock, check the corresponding log files in the running dataset/baseline. If nothing has been logged for some time and it does not say that the experiment has been completed, check the CPU utilization and DRAM usage. It is likely a DRAM out-of-memory problem. The experiments would likely take up more DRAM. If a larger machine is unavailable, try disabling (commenting out) `Muffliato` experiments in the run scripts.
- If you get a `ModuleNotFoundError`, verify the conda environment `venv` is active and you followed the steps in the [Setting up the Environment Section](#set-up-the-environment-only-for-functional-and-reproduced-badges).

#### Copying results back from Docker
We provided `docker-copy-exp-1.sh` and `docker-copy-exp-2.sh` to copy the results from the docker containers to the subfolders.
Experiment 1 (Gradient inversion results) are copied into ```$SHATTER_HOME/artifact_scripts/gradientInversion/rog/experiments```.
Experiment 2 (Convergence, MIA, LA) are copied into ```artifact_scripts/small_scale/results```.


## Limitations (Only for Functional and Reproduced badges)
- The full results are not reproducible without 25 machines and a long run time, therefore we provided smaller scale experiments for the functional badge. The code works for a multi-machine setup similar to DecentralizePy.
- Attacks, especially linkability attack here are much more powerful here than in the main results of the paper since it is much easier to attack at such a small scale of 8 clients and Shatter with only 2 virtual nodes does not make a lot of difference.


## Notes on Reusability (Only for Functional and Reproduced badges)
The code for Shatter is written in the same structure as DecentralizePy, so it can easily be plugged into projects using DecentralizePy.
Furthermore, the code for the `attacks` in Shatter can be used beyond the scope of Shatter for general privacy-preserving research.
To add more datasets, one needs to add the dataset in `decentralizepy.datasets` and the code to execute attacks in `virtualNodes.attacks`.
Config files provided in the baselines are self explanatory to modify parameters. Please create an issue in the repository if some parameter is difficult to understand.
In gradient inversion experiments, only 1 client was attacked across all baselines. To run a full version, set the `num_clients=100` in `artifact_scripts/gradientInversion/rog/run.sh`. This should take about 150x time and space. This can also be parallelized to fill up the available GPU memory since clients are independent.

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
No special hardware needed.

### Software Requirements
We tested the artifacts on Ubuntu 22.04 and Python 3.10. This should however not be a strict requirement. Dependencies can be installed through `requirements.txt`.
Git-LFS is essential to download the datasets and the models.
Use `git lfs pull` to ensure large files are downloaded after cloning.

### Estimated Time and Storage Consumption
Each experiment should take roughly an hour. So, in total, the experiments should take ~5.5 hours.
Each experiment should take up roughly 200MB of storage, totalling up to ~4GBs of storage including the datasets and the models (used for eval).
All time measurements were done when using 1 CPU and 1 Nvidia A100 80GB GPU.

## Environment

### Accessibility (All badges)
The artifact code and data can be accessed via [https://github.com/sacs-epfl/shatter](https://github.com/sacs-epfl/shatter). This is the lab's public Github.
We also provide a Dockerfile in the repository and the docker image on the Docker Hub as `rishis8/shatter-artifact-pets2025:latest`.
When cloning directly from the Github repository, git-lfs is required to download the datasets and models.

### Set up the environment (Only for Functional and Reproduced badges)
We recommend using the Docker image (`rishis8/shatter-artifact-pets2025:latest`) since everything is already set up. In all further instructions, we assume that the default directory is the `shatter` repository root. In the docker image, this is `/root/shatter`.
The only thing needed is to install rog.
```bash
cd artifact_scripts/gradientInversion/rog
conda activate venv
pip install -r requirements.txt
```

If not using docker, **please create a virtual environment with python 3.10** and use the following commands in the root directory of the repository:
```bash
pip install -r requirements.txt
cd artifact_scripts/gradientInversion/rog
sudo apt-get install -y libgl1-mesa-glx
pip install -r requirements.txt
unzip data.zip
rm data.zip
```

### Testing the Environment (Only for Functional and Reproduced badges)
Check if CUDA is available, and `transformers` and `torch` can be imported in python venv.
Check `artifact_scripts/gradientInversion/rog/data/val` and `artifact_scripts/gradientInversion/rog/model_zoos` exist.
Finally, check `eval/data/CIFAR10`, `eval/data/movielens`, and `eval/data/sent140` exist.

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
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

#### Experiment 1: Gradient-inversion attack
- Inside the docker image or the correct environment setup, change the working directory to `artifact_scripts/gradientInversion/rog`.
- Run `./run.sh ~/.conda/envs/venv/bin`. This should take ~15 minutes and about 30 MBs of space because of reconstructed images.
- Reconstructed images per client, aggregated data CSVs and bar plots are generated in `artifact_scripts/gradientInversion/rog/experiments/lenet`.
- VNodes{k} is Shatter with k virtual nodes.
- The images and lpips scores can be compared to Figures 2 and 8. These will not be exact numbers since only 1 client was attacked as opposed to 100 in the experiments in the paper.
- We recommend clearing up `artifact_scripts/gradientInversion/rog/experiments/lenet` before running other experiments to save disk space.
- Next change the working directory back to the root directory of `shatter` repository. In the docker image, this is `/root/shatter`.

#### Experiment 2: Convergence, MIA and LA
- Inside the docker image or the correct environment setup, change the working directory to `artifact_scripts/small_scale`.
- These are smaller scale versions of the other experiments in the paper since the full-scale experiments take very long and need to be run across 25 machines.
- Quickest way is to perform `./run_all /root/shatter/ .conda/envs/venv/bin`. Here, `/root/shatter` is the absolute path to the `shatter` repository (defailt in the docker image). This runs the experiments for all the datasets in one go. To do this step by step, one can also individually run the scripts for each dataset in `artifact_scripts/small_scale` with the same command line arguments. Experiments with CIFAR-10 and Movielens datasets should take ~1.5 hour and ~200MBs in disk space each. Twitter dataset experiments take a bit longer and can take ~2.5 hours and ~200 MBs. In total `run_all` should run in ~5.5 hours and ~600MBs of disk space.
- Inside `artifact_scripts/small_scale/CIFAR10`, for each baseline, a new folder will be created for the experiment. The aggregated CSVs with `*test_acc.csv` (Figure 3, 5, 7 all except Movielens), `*test_loss.csv` (Figure 3, 5, 7 Movielens), `*clients_linkability.csv` (Figure 6), `*clients_MIA.csv` (Figure 6), `*iterations_linkability.csv` (Partially Figure 7c), and `*iterations_MIA.csv` (Figure 5). PDFs for plots (not exactly the ones in the paper) are also created in the same folders. Since these are smaller scale experiments, the values will not match the ones in the paper.
- For experiments with Muffliato, since we only train once every 10 rounds, the iterations axis is stretched to 10x in the CSVs and plots. This is corrected in the results in the main paper post-hoc.
- If CUDA OOM is encountered, try lowering the `test_batch_size` and `batch_size` in `config*.ini` within each dataset and baseline folder. One such `config` file is `artifact_scripts/small_scale/CIFAR10/EL/config_EL.ini`
- If the experiments look like they are in a deadlock, check the corresponding log files in the running dataset/baseline. If nothing has been logged for some time and it does not say that the experiment has been completed, check the CPU utilization and DRAM usage. It is likely a DRAM out-of-memory problem. The experiments would likely take up more DRAM. If a larger machine is unavailable, try disabling (commenting out) `Muffliato` experiments in the run scripts.


## Limitations (Only for Functional and Reproduced badges)
- The full results are not reproducible without 25 machines and a long run time, therefore we provided smaller scale experiments for the functional badge. The code works for a multi-machine setup similar to DecentralizePy.
- Attacks, especially linkability attack here are much more powerful here than in the main results of the paper since it is much easier to attack at such a small scale of 8 clients and Shatter with only 2 virtual nodes does not make a lot of difference.


## Notes on Reusability (Only for Functional and Reproduced badges)
The code for Shatter is written in the same structure as DecentralizePy, so it can easily be plugged into projects using DecentralizePy.
Furthermore, the code for the `attacks` in Shatter can be used beyond the scope of Shatter for general privacy-preserving research.
To add more datasets, one needs to add the dataset in `decentralizepy.datasets` and the code to execute attacks in `virtualNodes.attacks`.
Config files provided in the baselines are self explanatory to modify parameters. Please create an issue in the repository if some parameter is difficult to understand.
In gradient inversion experiments, only 1 client was attacked across all baselines. To run a full version, set the `num_clients=100` in `artifact_scripts/gradientInversion/rog/run.sh`. This should take about 150x time and space. This can also be parallelized to fill up the available GPU memory since clients are independent.

import logging
import os
from pathlib import Path
from shutil import copy

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from virtualNodes.mappings.VNodeLinear import VNodeLinear
from virtualNodes.node.dropouts.VNodePeerSampler import VNodePeerSampler
from virtualNodes.node.dropouts.VNodeRealSimulation import VNodeReal


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]
    m_id = args.machine_id

    sm = args.server_machine
    sr = args.server_rank

    l = VNodeLinear(
        n_machines, procs_per_machine, global_service_machine=sm, current_machine=m_id
    )

    processes = []
    if sm == m_id:
        processes.append(
            mp.Process(
                target=VNodePeerSampler,
                args=[
                    sr,
                    m_id,
                    l,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    log_level[args.log_level],
                ],
            )
        )

    for r in range(m_id * procs_per_machine, (m_id + 1) * procs_per_machine):
        processes.append(
            mp.Process(
                target=VNodeReal,
                args=[
                    r,
                    m_id,
                    l,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    args.weights_store_dir,
                    log_level[args.log_level],
                    args.test_after,
                    args.train_evaluate_after,
                    args.reset_optimizer,
                    l.get_uid(sr, sm),
                ],
                daemon=False,
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()

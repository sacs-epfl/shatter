import logging
import os

import torch

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.utils import write_results_to_csv
from virtualNodes.node.VNode import VNode


class VNodeFake(VNode):
    """
    This class defines the node for decentralized learning with virtual nodes.

    """

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        return (
            self.messages_received[self.iteration] == len(self.my_neighbors) + 1
        )  # + 1 for the initial message from the master node

    def run(self):
        """
        Start the decentralized learning

        """

        self.messages_received = [0 for _ in range(self.iterations)]

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)

            self.iteration = iteration

            new_neighbors = self.get_neighbors()

            self.my_neighbors = new_neighbors
            self.connect_neighbors()

            logging.debug("Connected to all neighbors")

            while not self.received_from_all():
                sender, data = self.receive_Virtual()

                logging.debug(
                    "Received a Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
                )

                if sender == self.master_node:
                    # To forward to neighbors
                    for neighbor in self.my_neighbors:
                        self.communication.send(neighbor, data)
                else:
                    # To send to server
                    data["vSource"] = sender
                    self.communication.send(self.master_node, data)
                self.messages_received[data["iteration"]] += 1

            logging.info("Received all messages for iteration {}".format(iteration))
            self.communication.send(
                self.master_node,
                {
                    "iteration": iteration,
                    "CHANNEL": "VNodeDPSGD",
                    "ROUND_COMPLETE": True,
                },
            )

            results_dict = {
                "iteration": iteration + 1,
                "total_bytes": self.communication.total_bytes,
                "total_meta": (
                    self.communication.total_meta
                    if hasattr(self.communication, "total_meta")
                    else None
                ),
                "total_data_per_n": (
                    self.communication.total_data
                    if hasattr(self.communication, "total_data")
                    else None
                ),
            }
            write_results_to_csv(
                os.path.join(self.log_dir, "{}_results.csv".format(self.rank)),
                results_dict,
            )

        self.disconnect_neighbors()

        logging.info("All neighbors disconnected. Process complete!")

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        peer_sampler_uid=-1,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        self.init_log(log_dir, rank, log_level)
        logging.info("Started process.")

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )

        nodeConfigs = config["NODE"]
        self.vnodes_per_node = nodeConfigs["vnodes_per_node"]

        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()

        self.master_node = self.get_master_node()
        self.peer_sampler_uid = peer_sampler_uid

        self.connect_neighbor(self.master_node)
        self.connect_neighbor(self.peer_sampler_uid)
        self.wait_for_hello(self.master_node)
        self.wait_for_hello(self.peer_sampler_uid)

        # self.connect_neighbors()

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        peer_sampler_uid=-1,
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            None,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            peer_sampler_uid,
            *args
        )
        self.run()

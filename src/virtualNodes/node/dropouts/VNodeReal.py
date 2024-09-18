import random
import copy
import logging
import math
import os
from collections import deque
from time import perf_counter

import torch
from torch import multiprocessing as mp

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.utils import write_results_to_csv
from virtualNodes.node.VNode import VNode
from virtualNodes.node.VNodeFake import VNodeFake


class VNodeReal(VNode):
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

        return self.round_complete[self.iteration] == self.vnodes_per_node
    
    def participate(self):
        """
        Participate in the current round

        """

        if self.crash_real:
            if self.crashed_last_round:
                to_crash = self.crash_rng.random() < (self.crash_probability + self.crash_correlation * (1 - self.crash_probability))
            else:
                to_crash = self.crash_rng.random() < self.crash_probability

            return not to_crash
    
        return True


    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        prev_elapsed_time = 0

        self.round_complete = [0 for _ in range(self.iterations)]

        self.crashed_last_round = False
        self.crash_rng = random.Random()
        self.crash_rng.seed(self.dataset.random_seed * 125 + self.uid)

        # Perturb the model parameters
        if self.perturb_model:
            logging.info("Perturbing model")
            state_dict = self.model.state_dict()
            rng = torch.Generator()
            rng.manual_seed(self.dataset.random_seed * 125 + self.uid)
            for key, value in state_dict.items():
                noise = (
                    torch.rand(value.shape, generator=rng) / 100 - 0.005
                ) * self.perturb_multiplier
                state_dict[key] = value + noise
            self.model.load_state_dict(state_dict)

        self.sharing.copy_model(self.model)

        for iteration in range(self.iterations):

            # Learning rate decay
            if iteration > 0 and iteration % self.reduce_lr_after == 0:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * self.reduce_by

            start_time = perf_counter()
            train_time = 0
            eval_time = 0
            agg_time = 0
            total_time_no_eval = 0
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            to_participate = self.participate()
            
            if to_participate:
                logging.info("Training this round.")
                self.trainer.train(self.dataset)

            if self.log_models and (iteration == 0 or rounds_to_train_evaluate == 0):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.weights_store_dir,
                        "{}_{}_inter.pt".format(self.uid, iteration),
                    ),
                )

            agg_start_time = perf_counter()
            train_time = agg_start_time - start_time


            if to_participate:
                to_send_list = self.sharing.get_data_to_send(
                    vnodes_per_node=self.vnodes_per_node, sparsity=self.sparsity
                )
                self.crashed_last_round = False
            else:
                self.crashed_last_round = True
                to_send_list = [{"NOT_WORKING": True} for _ in range(self.vnodes_per_node)]


            for i, to_send in enumerate(to_send_list):
                to_send["CHANNEL"] = "VNodeDPSGD"
                to_send["iteration"] = self.iteration
                self.communication.send(self.vids[i], to_send)
                logging.debug(
                    "Sending message to {} of iteration {}".format(
                        self.vids[i], self.iteration
                    )
                )
            del to_send_list

            received_at_least_once = False

            while not self.received_from_all():
                forwarder, data = self.receive_Virtual()
                assert forwarder in self.vids

                if "ROUND_COMPLETE" in data:
                    logging.debug(
                        "Received ROUND_COMPLETE from {} for iteration {}".format(
                            forwarder, iteration
                        )
                    )
                    self.round_complete[data["iteration"]] += 1
                    continue

                logging.debug(
                    "Received Forwarder Model from {} of iteration {}".format(
                        forwarder, data["iteration"]
                    )
                )

                sender = data["vSource"]

                if data["iteration"] == self.iteration and "NOT_WORKING" not in data and to_participate:
                    received_at_least_once = True
                    self.sharing.forward_averaging(data)
                elif "NOT_WORKING" not in data:
                    if sender not in self.peer_deques:
                        self.peer_deques[sender] = deque()
                    self.peer_deques[sender].append(data)
                else:
                    logging.info("Virtual node {} of Real node {} is not working".format(sender, self.get_master_node(sender)))

            averaging_deque = dict()
            for neighbor in self.peer_deques:
                averaging_deque[neighbor] = deque()
                for deq in self.peer_deques[neighbor]:
                    if deq["iteration"] == self.iteration:
                        received_at_least_once = True
                        averaging_deque[neighbor].append(deq)
                        logging.debug(
                            "Virtual node {} has sent the current iteration".format(
                                neighbor
                            )
                        )
                for deq in averaging_deque[neighbor]:
                    self.peer_deques[neighbor].remove(deq)

            if received_at_least_once and to_participate:
                self.sharing.finish_forward_averaging(averaging_deque)
            del averaging_deque

            # if iteration == 0 or rounds_to_train_evaluate == 0:
            #     torch.save(
            #         self.model.state_dict(),
            #         os.path.join(
            #             self.weights_store_dir,
            #             "{}_{}_post.pt".format(self.uid, iteration),
            #         ),
            #     )

            agg_end_time = perf_counter()
            agg_time = agg_end_time - agg_start_time
            total_time_no_eval = agg_end_time - start_time

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            results_dict = {
                "iteration": iteration + 1,
                "train_loss": None,
                "test_loss": None,
                "test_acc": None,
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
                "agg_time": agg_time,
                "train_time": train_time,
                "eval_time": None,
                "total_round_time_no_eval": total_time_no_eval,
                "total_elapsed_time_no_eval": total_time_no_eval + prev_elapsed_time,
            }

            prev_elapsed_time += total_time_no_eval

            if iteration == 0 or iteration % self.train_evaluate_after == 0:
                logging.info("Evaluating on train set.")
                # rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"] = loss_after_sharing

            if self.dataset.__testing__ and iteration % self.test_after == 0:
                eval_start_time = perf_counter()
                # rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                eval_time = perf_counter() - eval_start_time
                results_dict["test_acc"] = ta
                results_dict["test_loss"] = tl

                # if global_epoch == 49:
                #     change *= 2

                # global_epoch += change

            results_dict["eval_time"] = eval_time

            write_results_to_csv(
                os.path.join(self.log_dir, "{}_results.csv".format(self.rank)),
                results_dict,
            )

        self.disconnect_neighbors()

        for p in self.virtualProcs:
            p.join()

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
        self.log_models = (
            nodeConfigs["log_models"] if "log_models" in nodeConfigs else False
        )
        self.perturb_model = (
            nodeConfigs["perturb_model"] if "perturb_model" in nodeConfigs else False
        )
        self.perturb_multiplier = (
            nodeConfigs["perturb_multiplier"]
            if "perturb_multiplier" in nodeConfigs
            else 1
        )
        self.vids = self.get_vids()
        self.reduce_lr_after = (
            nodeConfigs["reduce_lr_after"] if "reduce_lr_after" in nodeConfigs else 10e9
        )
        self.reduce_by = nodeConfigs["reduce_by"] if "reduce_by" in nodeConfigs else 0.1
        self.sparsity = nodeConfigs["sparsity"] if "sparsity" in nodeConfigs else 0.0
        threads_per_proc = (
            nodeConfigs["threads_per_proc"] if "threads_per_proc" in nodeConfigs else 1
        )


        crashConfigs = config["CRASH"] if "CRASH" in config else dict()
        self.crash_real = crashConfigs["crash_real"] if "crash_real" in crashConfigs else False
        self.crash_probability = crashConfigs["crash_probability"] if "crash_probability" in crashConfigs else 0.0
        self.crash_correlation = crashConfigs["crash_correlation"] if "crash_correlation" in crashConfigs else 0.0


        torch.set_num_threads(threads_per_proc)
        torch.set_num_interop_threads(1)

        logging.info(
            "Each proc uses %d threads out of %d.", threads_per_proc, os.cpu_count()
        )

        # Start the virtual process
        self.virtualProcs = []
        for vrank in self.vids:
            self.virtualProcs.append(
                mp.Process(
                    target=VNodeFake,
                    args=[
                        vrank,
                        machine_id,
                        mapping,
                        config,
                        iterations,
                        log_dir,
                        weights_store_dir,
                        log_level,
                        test_after,
                        train_evaluate_after,
                        reset_optimizer,
                        *args,
                    ],
                    daemon=False,
                )
            )

        for i, p in enumerate(self.virtualProcs):
            p.start()

        self.init_log(log_dir, rank, log_level)
        logging.info("Started process.")

        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.vids
        self.peer_sampler_uid = peer_sampler_uid

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()

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
        # total_threads = os.cpu_count()
        # self.threads_per_proc = 4
        # self.threads_per_proc = max(
        #     math.floor(total_threads / mapping.get_local_procs_count()), 1
        # )
        # torch.set_num_threads(self.threads_per_proc)
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
        # logging.info(
        #     "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        # )
        self.run()

import copy
import json
import logging
import os
import sys

import torch
from torch.nn import CrossEntropyLoss, MSELoss

from decentralizepy.datasets.CIFAR10 import CIFAR10
from decentralizepy.datasets.MovieLens import MovieLens
from virtualNodes.attacks.LinkabilityAttack import LinkabilityAttack
from virtualNodes.sharing.VNodeSharing import VNodeSharing


class VNodeSharingAttack(VNodeSharing):
    """
    Sharing class for Virtual Nodes.

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
        attack_after=8,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet!
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            compress,
            compression_package,
            compression_class,
            float_precision,
        )
        self.attack_after = attack_after
        trainset_dict = dict()
        self.num_clients = self.mapping.get_n_procs()

        for client in range(self.num_clients):
            torch.manual_seed(self.dataset.random_seed)
            c_rank, c_machine_id = self.mapping.get_machine_and_rank(client)
            assert self.dataset.only_local == False
            if isinstance(self.dataset, CIFAR10):
                # Loads all the data
                this_trainset = self.dataset.get_trainset(
                    batch_size=self.dataset.test_batch_size,
                    shuffle=False,
                    dataset_id=client,
                )
            else:
                # Each one loads data of only 1 client
                this_trainset = type(self.dataset)(
                    rank=c_rank,
                    machine_id=c_machine_id,
                    mapping=self.dataset.mapping,
                    random_seed=self.dataset.random_seed,
                    only_local=self.dataset.only_local,
                    train_dir=self.dataset.train_dir,
                    test_dir="",
                    sizes="",
                    test_batch_size=self.dataset.test_batch_size,
                    partition_niid=self.dataset.partition_niid,
                    alpha=self.dataset.alpha,
                    shards=self.dataset.shards,
                    validation_source=self.dataset.validation_source,
                    validation_size=self.dataset.validation_size,
                ).get_trainset(batch_size=self.dataset.test_batch_size, shuffle=False)
            trainset_dict[client] = this_trainset

        loss = MSELoss() if isinstance(self.dataset, MovieLens) else CrossEntropyLoss()

        self.attacker = LinkabilityAttack(
            self.num_clients,
            trainset_dict,
            loss,
        )
        self.attack_results = dict()
        self.attack_model = copy.deepcopy(self.model)
        self.prev_avg_model = copy.deepcopy(self.model)

    def __del__(self):
        # Write the attack results to a file
        with open(
            os.path.join(self.log_dir, "attacker_{}.json".format(self.uid)), "w"
        ) as f:
            json.dump(self.attack_results, f)

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            weights = torch.zeros(
                self.total_length + 1, dtype=torch.int32
            )  # Add an index at the end
            weights[0] = 1
            weights[-1] = -1

            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)

            T = torch.cat(tensors_to_cat, dim=0)
            if self.communication_round % self.attack_after == 0:
                tensors_to_cat = []
                for _, v in self.prev_avg_model.state_dict().items():
                    t = v.flatten()
                    tensors_to_cat.append(t)
                T2 = torch.cat(tensors_to_cat, dim=0)

            for _, n in enumerate(peer_deques):
                for data in peer_deques[n]:
                    iteration = data["iteration"]
                    correct_real_node = data["real_node"]
                    if "degree" in data:
                        del data["degree"]
                    del data["iteration"]
                    del data["CHANNEL"]
                    del data["real_node"]

                    logging.debug(
                        "Averaging model from neighbor {} of iteration {}".format(
                            n, iteration
                        )
                    )
                    deserializedT, start, end = self.deserialized_model(data)
                    logging.debug("Deserialized model from neighbor {}".format(n))
                    T += deserializedT
                    if (
                        self.communication_round % self.attack_after == 0
                        and correct_real_node != self.uid
                    ):
                        logging.info("Attacking neighbor {}".format(n))
                        T1 = copy.deepcopy(T2)
                        T1[start:end] = deserializedT[start:end]
                        T1_state_dict = self._post_step(T1)
                        self.attack_model.load_state_dict(T1_state_dict)
                        self.attack_model.eval()
                        predicted_client = self.attacker.attack(
                            self.attack_model, skip=[self.uid]
                        )
                        logging.info(
                            "Original client: {}, Predicted as: {}".format(
                                correct_real_node, predicted_client
                            )
                        )
                        # Fix getting twice from the same correct_client in the same iteration
                        if correct_real_node not in self.attack_results:
                            self.attack_results[correct_real_node] = dict()
                        self.attack_results[correct_real_node][
                            self.communication_round
                        ] = predicted_client

                    # Prefix sum weight trick
                    weights[start] += 1
                    weights[end] -= 1

            # Prefix sum trick continued
            weights = torch.cumsum(weights, dim=0)[
                :-1
            ]  # Remove the previously added index
            weights = weights.type(torch.float32)
            weights = 1.0 / weights
            T = T * weights

            logging.debug("Finished averaging")

            state_dict = self._post_step(T)
            self.model.load_state_dict(state_dict)
            self.prev_avg_model.load_state_dict(state_dict)
            self.communication_round += 1

    def get_data_to_send(self, vnodes_per_node=1, degree=None):
        self._pre_step()
        data_list = self.serialized_models(vnodes_per_node=vnodes_per_node)
        for data in data_list:
            data["real_node"] = self.uid
        return data_list

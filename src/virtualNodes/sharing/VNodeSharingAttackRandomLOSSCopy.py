import copy
import json
import logging
import os
import sys
from time import sleep

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

from decentralizepy.datasets.CIFAR10 import CIFAR10
from decentralizepy.datasets.MovieLens import MovieLens
from decentralizepy.datasets.text.Twitter import Twitter
from virtualNodes.attacks.LinkabilityAttack import LinkabilityAttack
from virtualNodes.attacks.LinkabilityAttackTwitter import LinkabilityAttackTwitter
from virtualNodes.attacks.MIA import LOSSCIFAR_ResNET, LOSSTwitter
from virtualNodes.sharing.VNodeSharingRandom import VNodeSharing


class VNodeSharingAttackRandomLOSS(VNodeSharing):
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
        perform_attack=True,
        attack_random=8,
        will_receive=8,
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

        print(
            "Total length: {} | {}".format(self.total_length, type(self.total_length))
        )

        self.attack_after = attack_after
        self.perform_attack = perform_attack
        self.attack_random = attack_random
        self.will_receive = will_receive
        trainset_dict = dict()
        self.num_clients = self.mapping.get_n_procs()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.current_weights = None
        self.current_sum = None

        # instantiate a torch generator
        self.random_indices = None

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
                self.linkabilityAttack = LinkabilityAttack
                self.mia = LOSSCIFAR_ResNET.LOSSMIA
            elif isinstance(self.dataset, Twitter):
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
                    tokenizer="BERT",
                    at_most=self.dataset.at_most,
                ).get_trainset(batch_size=self.dataset.test_batch_size, shuffle=False)
                self.linkabilityAttack = LinkabilityAttackTwitter
                self.mia = LOSSTwitter.LOSSMIA
            trainset_dict[client] = this_trainset
        loss = MSELoss() if isinstance(self.dataset, MovieLens) else CrossEntropyLoss()

        self.linkabilityAttack = self.linkabilityAttack(
            self.num_clients,
            trainset_dict,
            loss,
        )
        self.attack_results = {
            "linkability": {},
            "loss_vals": {},
        }

        self.attack_model = None
        self.attack_random_generator = torch.Generator()
        self.attack_random_generator.manual_seed(
            self.dataset.random_seed * 100 + self.uid
        )
        self.attack_counter = 0

        self.seed = self.dataset.random_seed
        self.train_dir = self.dataset.train_dir
        self.mia = self.mia(self.train_dir, self.dataset.test_batch_size)

    def copy_model(self, model):
        """
        Copies the model

        Parameters
        ----------
        model : torch.nn.Module
            Model to copy

        """

        self.attack_model = copy.deepcopy(model)
        tensors_to_cat = []
        for _, v in self.attack_model.state_dict().items():
            t = v.flatten()
            tensors_to_cat.append(t)
        self.T = torch.cat(tensors_to_cat, dim=0).to(self.device)

    def __del__(self):
        # Write the attack results to a file
        if self.perform_attack:
            torch.save(
                self.attack_results,
                os.path.join(self.log_dir, "{}_attacker.pth".format(self.uid)),
            )

    def forward_averaging(self, data):
        """
        Computes the sum for the average in a state based manner.

        Parameters
        ----------
        data : dict
            Received data

        Returns
        -------
        None

        """
        with torch.no_grad():
            if self.current_sum == None:
                # First time take model of self

                if self.attack_model == None:
                    copy.deepcopy(self.model)

                self.current_weights = (
                    torch.zeros(
                        self.total_length, dtype=torch.float32, device=self.device
                    )
                    + 1
                )

                tensors_to_cat = []
                for _, v in self.model.state_dict().items():
                    t = v.flatten()
                    tensors_to_cat.append(t)
                self.current_sum = torch.cat(tensors_to_cat, dim=0).to(self.device)
                if (
                    self.perform_attack
                    and self.communication_round % self.attack_after == 0
                ):
                    self.attack_counter = 0
                    self.to_attack_this_round = torch.zeros(
                        (self.will_receive,), dtype=torch.bool
                    )
                    attacking_indices = torch.randperm(
                        self.will_receive, generator=self.attack_random_generator
                    )[: self.attack_random]
                    self.to_attack_this_round[attacking_indices] = True

            iteration = data["iteration"]
            correct_real_node = data["real_node"]
            not_trained = False
            if "degree" in data:
                del data["degree"]
            if "not_trained" in data:
                not_trained = data["not_trained"]
                del data["not_trained"]
            del data["iteration"]
            del data["CHANNEL"]
            logging.debug(
                "Forward Averaging model from neighbor {} of iteration {}".format(
                    data["vSource"], iteration
                )
            )
            try:
                deserializedT, indices = self.deserialized_model(data)
            except Exception as e:
                print("uid: {} | Exception: {}".format(self.uid, e))
                raise e
            logging.debug("Deserialized model from neighbor {}".format(data["vSource"]))

            deserializedT = deserializedT.to(self.device)

            self.current_sum[indices] += deserializedT
            self.current_weights[indices] += 1

            # Averaging done

            if (
                self.perform_attack
                and (not not_trained)
                and self.communication_round % self.attack_after == 0
                and correct_real_node != self.uid
                and self.to_attack_this_round[self.attack_counter]
            ):
                # Complete the chunk with deserializedT and attack
                T2 = copy.deepcopy(self.T).to(self.device)
                T2[indices] = deserializedT
                T2_state_dict = self._post_step(T2)
                self.attack_model.load_state_dict(T2_state_dict)
                self.attack_model.eval()
                # Linkability
                logging.info("Linking neighbor {}".format(data["vSource"]))
                predicted_client = self.linkabilityAttack.attack(
                    self.attack_model, skip=[self.uid]
                )
                logging.info(
                    "Original client: {}, Linked as: {}".format(
                        correct_real_node, predicted_client
                    )
                )
                if correct_real_node not in self.attack_results["linkability"]:
                    self.attack_results["linkability"][correct_real_node] = dict()
                if (
                    self.communication_round
                    not in self.attack_results["linkability"][correct_real_node]
                ):
                    self.attack_results["linkability"][correct_real_node][
                        self.communication_round
                    ] = []
                self.attack_results["linkability"][correct_real_node][
                    self.communication_round
                ].append(predicted_client)

                # LiRA-LOSS
                logging.info("MIA on neighbor {}".format(data["vSource"]))
                loss_vals = self.mia.attack_dataset(self.attack_model)

                if correct_real_node not in self.attack_results["loss_vals"]:
                    self.attack_results["loss_vals"][correct_real_node] = dict()

                if (
                    self.communication_round
                    not in self.attack_results["loss_vals"][correct_real_node]
                ):
                    self.attack_results["loss_vals"][correct_real_node][
                        self.communication_round
                    ] = []

                self.attack_results["loss_vals"][correct_real_node][
                    self.communication_round
                ].append(loss_vals.cpu())
                torch.save(
                    self.attack_results,
                    os.path.join(self.log_dir, "{}_attacker.pth".format(self.uid)),
                )

            self.attack_counter += 1

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """
        with torch.no_grad():
            for _, n in enumerate(peer_deques):
                for data in peer_deques[n]:
                    self.forward_averaging(data)

            assert self.current_sum != None
            assert self.current_weights != None

            self.current_weights = self.current_weights.type(torch.float32)
            self.current_weights = 1.0 / self.current_weights
            self.current_sum = self.current_sum * self.current_weights
            logging.debug("Finished averaging")
            self.T = self.current_sum
            self.current_sum = self.current_sum.cpu()
            self.model.load_state_dict(self._post_step(self.current_sum))
            self.communication_round += 1
            self.current_weights = None
            self.current_sum = None

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        raise NotImplementedError()

    def get_data_to_send(self, vnodes_per_node=1, degree=None):
        self._pre_step()
        data_list = self.serialized_models(vnodes_per_node=vnodes_per_node)
        for data in data_list:
            data["real_node"] = self.uid
        return data_list

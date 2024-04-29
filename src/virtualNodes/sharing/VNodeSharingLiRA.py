import copy
import logging
import os

import torch
from torch.utils.data import DataLoader

from virtualNodes.attacks.MIA.LiRACIFAR import LiRACIFAR10, LiRAMIA
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
        attack_after=20,
        shadow_weights_store_dir="/mnt/nfs/risharma/Gitlab/virtualNodes/src/virtualNodes/attacks/MIA/weights_fixed_256",
        shadow_model_confidence_path="/mnt/nfs/risharma/Gitlab/virtualNodes/src/virtualNodes/attacks/MIA/weights_fixed_256/confidences_clamped_seed91_K256.pt",
        attack_batch_size=256,
        K=256,
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
        self.num_clients = self.mapping.get_n_procs()

        self.K = K
        self.attack_batch_size = attack_batch_size
        self.seed = self.dataset.random_seed
        self.train_dir = self.dataset.train_dir
        self.shadow_weights_store_dir = shadow_weights_store_dir
        self.shadow_model_confidence_path = shadow_model_confidence_path
        self.shadow_dataset_model = LiRACIFAR10(
            K=self.K, random_seed=self.seed, train_dir=self.train_dir
        )
        self.mia = LiRAMIA(
            self.shadow_dataset_model, weights_store_dir=self.shadow_weights_store_dir
        )
        self.attack_dataloader = DataLoader(
            self.shadow_dataset_model.trainset,
            batch_size=self.attack_batch_size,
            shuffle=False,
        )
        if os.path.isfile(self.shadow_model_confidence_path):
            self.mia.load_shadow_confidences(self.shadow_model_confidence_path)
        else:
            logging.info("Shadow model confidences not found. Computing them.")
            self.mia.precompute_shadow_confidences(batch_size=self.attack_batch_size)

        self.attack_results = {"lira_offline": {}, "lira_online": {}, "loss_vals": {}}
        self.attack_model = copy.deepcopy(self.model)
        self.prev_avg_model = copy.deepcopy(self.model)

    def __del__(self):
        # Write the attack results to a file
        torch.save(
            self.attack_results,
            os.path.join(self.log_dir, "mia_attacker_{}.pth".format(self.uid)),
        )

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

                        lir_offline, lir_online, loss_vals = self.mia.attack_dataset(
                            self.attack_model,
                            self.attack_dataloader,
                            len(self.shadow_dataset_model.trainset),
                            online=True,
                            return_loss=True,
                            return_both=True,
                        )

                        logging.info(
                            "Attacking client: {} complete.".format(correct_real_node)
                        )
                        if correct_real_node not in self.attack_results["lira_offline"]:
                            self.attack_results["lira_offline"][correct_real_node] = {}
                            self.attack_results["lira_online"][correct_real_node] = {}
                            self.attack_results["loss_vals"][correct_real_node] = {}
                        if (
                            self.communication_round
                            not in self.attack_results["lira_offline"][
                                correct_real_node
                            ]
                        ):
                            self.attack_results["lira_offline"][correct_real_node][
                                self.communication_round
                            ] = []
                            self.attack_results["lira_online"][correct_real_node][
                                self.communication_round
                            ] = []
                            self.attack_results["loss_vals"][correct_real_node][
                                self.communication_round
                            ] = []

                        self.attack_results["lira_offline"][correct_real_node][
                            self.communication_round
                        ].append(lir_offline)
                        self.attack_results["lira_online"][correct_real_node][
                            self.communication_round
                        ].append(lir_online)
                        self.attack_results["loss_vals"][correct_real_node][
                            self.communication_round
                        ].append(loss_vals)

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

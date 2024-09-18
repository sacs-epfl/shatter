import logging
from time import sleep
import copy
import torch

from decentralizepy.sharing.Sharing import Sharing


class VNodeSharing(Sharing):
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.current_weights = None
        self.current_sum = None

        # instantiate a torch generator
        self.random_indices = None

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
        del self.attack_model

    def serialized_models(self, vnodes_per_node=1, sparsity=0.0):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        list(dict)
            Model converted to dict

        """

        if self.random_indices is None or sparsity != 0.0:
            torch_gen = torch.Generator()
            if sparsity == 0.0:
                torch_gen.manual_seed(self.dataset.random_seed)
            else:
                torch_gen.manual_seed(
                    self.dataset.random_seed * self.uid * 100 + self.communication_round
                )
            with torch.no_grad():
                self.random_indices = [[] for i in range(vnodes_per_node)]
                state_dict_index = 0
                for k, v in self.model.state_dict().items():
                    random_perm = (
                        torch.randperm(v.numel(), generator=torch_gen)
                        + state_dict_index
                    )
                    sizes = int((v.numel() // vnodes_per_node) * (1.0 - sparsity))
                    index = 0
                    for i in range(vnodes_per_node - 1):
                        self.random_indices[i].append(
                            random_perm[index : index + sizes]
                        )
                        index += sizes
                    # Add the last part
                    if sparsity == 0.0:
                        self.random_indices[-1].append(random_perm[index:])
                    else:
                        self.random_indices[-1].append(
                            random_perm[index : min(v.numel(), index + sizes)]
                        )
                    state_dict_index += v.numel()
                for i in range(vnodes_per_node):
                    self.random_indices[i] = torch.cat(self.random_indices[i], dim=0)

        to_cat = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                to_cat.append(t)
        flat = torch.cat(to_cat)
        sizes = flat.shape[0] // vnodes_per_node
        to_return = []
        for i in range(vnodes_per_node):
            data = dict()
            data["params"] = flat[self.random_indices[i]]
            data["start_index"] = i
            data["sparsity"] = sparsity
            data["random_generation_seed"] = (
                self.dataset.random_seed
                if sparsity == 0.0
                else self.dataset.random_seed * self.uid * 100
                + self.communication_round
            )
            data["vnodes_per_node"] = vnodes_per_node
            to_return.append(self.compress_data(data))

        return to_return

    def deserialized_model(self, m):
        """
        Convert received dict to a model vector.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        with torch.no_grad():
            m = self.decompress_data(m)
            sparsity = m["sparsity"]
            random_generation_seed = m["random_generation_seed"]
            vnodes_per_node = m["vnodes_per_node"]
            del m["sparsity"]
            del m["random_generation_seed"]
            del m["vnodes_per_node"]
            if sparsity == 0.0:
                assert self.random_indices is not None
                indices = self.random_indices[m["start_index"]]
            else:
                torch_gen = torch.Generator()
                torch_gen.manual_seed(random_generation_seed)
                with torch.no_grad():
                    random_indices = [[] for i in range(vnodes_per_node)]
                    state_dict_index = 0
                    for k, v in self.model.state_dict().items():
                        random_perm = (
                            torch.randperm(v.numel(), generator=torch_gen)
                            + state_dict_index
                        )
                        sizes = int((v.numel() // vnodes_per_node) * (1.0 - sparsity))
                        index = 0
                        for i in range(vnodes_per_node - 1):
                            random_indices[i].append(random_perm[index : index + sizes])
                            index += sizes
                        # Add the last part
                        random_indices[-1].append(
                            random_perm[index : min(v.numel(), index + sizes)]
                        )
                        state_dict_index += v.numel()
                    for i in range(vnodes_per_node):
                        random_indices[i] = torch.cat(random_indices[i], dim=0)
                indices = random_indices[m["start_index"]]

            des = m["params"].to(torch.float32)
            return des, indices

    def _post_step(self, T):
        """
        Return state_dict of model.

        Parameters
        ----------
        T : torch.Tensor
            Flat model vector

        Returns
        -------
        state_dict
            state_dict of model

        """
        state_dict = self.model.state_dict()
        start_index = 0
        for i, key in enumerate(state_dict):
            end_index = start_index + self.lens[i]
            state_dict[key] = T[start_index:end_index].reshape(self.shapes[i])
            start_index = end_index
        return state_dict

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
        if self.current_sum == None:
            # First time take model of self
            self.current_weights = (
                torch.zeros(self.total_length, dtype=torch.float32, device=self.device)
                + 1
            )

            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            self.current_sum = torch.cat(tensors_to_cat, dim=0).to(self.device)

        iteration = data["iteration"]
        if "degree" in data:
            del data["degree"]
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
        self.current_sum[indices] += deserializedT.to(self.device)
        self.current_weights[indices] += 1

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        self.current_weights = self.current_weights.type(torch.float32)
        self.current_weights = 1.0 / self.current_weights
        self.current_sum = self.current_sum * self.current_weights
        logging.debug("Finished averaging")
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

    def get_data_to_send(self, vnodes_per_node=1, degree=None, sparsity=0.0):
        self._pre_step()
        data_list = self.serialized_models(
            vnodes_per_node=vnodes_per_node, sparsity=sparsity
        )
        return data_list

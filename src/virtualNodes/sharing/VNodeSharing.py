import logging

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

    def serialized_models(self, vnodes_per_node=1):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        list(dict)
            Model converted to dict

        """
        to_cat = []
        with torch.no_grad():
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                to_cat.append(t)
        flat = torch.cat(to_cat)
        sizes = flat.shape[0] // vnodes_per_node
        index = 0
        to_return = []
        for _ in range(vnodes_per_node - 1):
            data = dict()
            data["params"] = flat[index : index + sizes]
            data["start_index"] = index
            index += sizes
            to_return.append(self.compress_data(data))

        # Add the last part
        data = dict()
        data["params"] = flat[index:]
        data["start_index"] = index
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
            start = m["start_index"]
            end = m["start_index"] + m["params"].shape[0]
            des = m["params"].to(torch.float32)
            return des, start, end

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
            self.current_weights = torch.zeros(
                self.total_length + 1, dtype=torch.float32, device=self.device
            )
            self.current_weights[0] = 1
            self.current_weights[-1] = -1

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
            deserializedT, start, end = self.deserialized_model(data)
        except Exception as e:
            print("uid: {} | Exception: {}".format(self.uid, e))
            raise e
        logging.debug("Deserialized model from neighbor {}".format(data["vSource"]))
        self.current_sum[start:end] += deserializedT.to(self.device)
        self.current_weights[start] += 1
        self.current_weights[end] -= 1

    def finish_forward_averaging(self, peer_deques):
        """
        Finishes the forward averaging.

        """
        for _, n in enumerate(peer_deques):
            for data in peer_deques[n]:
                self.forward_averaging(data)

        assert self.current_sum != None
        assert self.current_weights != None

        self.current_weights = torch.cumsum(self.current_weights, dim=0)[:-1]
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

            for _, n in enumerate(peer_deques):
                for data in peer_deques[n]:
                    iteration = data["iteration"]
                    if "degree" in data:
                        del data["degree"]
                    del data["iteration"]
                    del data["CHANNEL"]
                    logging.debug(
                        "Averaging model from neighbor {} of iteration {}".format(
                            n, iteration
                        )
                    )
                    if self.uid == 3:
                        print("Data: {}".format(data))
                        print("Size: {}", data["params"].shape)
                    try:

                        deserializedT, start, end = self.deserialized_model(data)
                    except Exception as e:
                        print("uid: {} | Exception: {}".format(self.uid, e))
                        raise e
                    logging.debug("Deserialized model from neighbor {}".format(n))
                    T[start:end] += deserializedT

                    logging.debug("Added to weights")

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
            self.communication_round += 1

    def get_data_to_send(self, vnodes_per_node=1, degree=None):
        self._pre_step()
        data_list = self.serialized_models(vnodes_per_node=vnodes_per_node)
        return data_list

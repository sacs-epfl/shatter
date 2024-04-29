import importlib
import logging

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.graphs.Regular import Regular
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.PeerSamplerDynamic import PeerSamplerDynamic


class VNodePeerSampler(PeerSamplerDynamic):
    """
    This class defines the peer sampling service

    """

    def fix_identifiers(self, neighbors):
        """
        Fix the identifiers of the neighbors to be consistent with the global graph

        Parameters
        ----------
        neighbors : list
            List of neighbors

        Returns
        -------
        list
            List of neighbors with fixed identifiers

        """

        return [(neighbor + self.n_procs_real) for neighbor in neighbors]

    def get_neighbors(self, node, iteration=None):
        if iteration != None and self.dynamic == True:
            if self.messages_received[iteration] == 0:
                logging.info(
                    "Generating new graph in iteration, self.iteration: {}, {}".format(
                        iteration, self.iteration
                    )
                )
                assert iteration == self.iteration + 1
                self.iteration = iteration
                self.graphs[iteration] = Regular(
                    self.n_procs_real * self.vnodes_per_node,
                    self.graph_degree,
                    seed=self.random_seed * 100000 + iteration,
                )
            assert iteration in self.graphs
            to_return = self.fix_identifiers(
                self.graphs[iteration].neighbors(node - self.n_procs_real)
            )
            self.messages_received[iteration] += 1
            if (
                self.messages_received[iteration]
                == self.n_procs_real * self.vnodes_per_node
            ):
                del self.graphs[iteration]
            return to_return
        else:
            return self.fix_identifiers(self.graph.neighbors(node - self.n_procs_real))

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        config,
        iterations=1,
        log_dir=".",
        log_level=logging.INFO,
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
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """

        self.iteration = -1
        self.graphs = dict()

        nodeConfigs = config["NODE"]
        self.vnodes_per_node = nodeConfigs["vnodes_per_node"]
        self.graph_degree = nodeConfigs["graph_degree"]
        self.dynamic = None
        if "dynamic" in nodeConfigs:
            self.dynamic = nodeConfigs["dynamic"]

        self.instantiate(
            rank,
            machine_id,
            mapping,
            None,
            config,
            iterations,
            log_dir,
            log_level,
            *args
        )

        self.graph = Regular(
            self.n_procs_real * self.vnodes_per_node,
            self.graph_degree,
            seed=self.random_seed,
        )

        self.n_procs_real = self.mapping.get_n_procs()
        self.messages_received = [0 for _ in range(iterations)]

        self.run()

        logging.info("Peer Sampler exiting")

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank,
            self.machine_id,
            self.mapping,
            self.vnodes_per_node,
            **comm_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        log_level=logging.INFO,
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
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
        )

        self.init_dataset_model(config["DATASET"])

        self.message_queue = dict()

        self.barrier = set()

        self.init_comm(config["COMMUNICATION"])
        self.n_procs_real = self.mapping.get_n_procs()
        self.my_neighbors = [
            i
            for i in range(
                self.n_procs_real, self.n_procs_real * (self.vnodes_per_node + 1)
            )
        ]
        self.connect_neighbors()

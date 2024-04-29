import torch
from torch.utils.data import DataLoader, Subset

from decentralizepy.datasets.Partitioner import DataPartitioner


class LiRAPartitioner(DataPartitioner):
    """
    Class to partition the dataset for the LiRA attack

    """

    def __init__(self, data, K=16, seed=91):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        K : int
            Number of partitions
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        data_len = len(data)

        indexes = torch.arange(data_len, dtype=torch.int32)
        rng = torch.Generator()
        rng.manual_seed(seed)
        keep = torch.rand((K, data_len), generator=rng)

        order = keep.argsort(0)

        keep = order < (K // 2)  # Each element goes into K // 2 partitions

        for i in range(K):
            self.partitions.append(indexes[keep[i]])

    def use(self, rank):
        """
        Get the partition for the process with the given `rank`

        Parameters
        ----------
        rank : int
            Rank of the current process

        Returns
        -------
        Partition
            The dataset partition of the current process

        """
        return Partition(self.data, self.partitions[rank])

    def get_partition_from_indices(self, indices):
        return Partition(self.data, indices)


class Partition(object):
    """
    Class for holding the data partition

    """

    def __init__(self, data, index):
        """
        Constructor. Caches the data and the indices

        Parameters
        ----------
        data : indexable
        index : list
            A list of indices

        """
        self.data = Subset(data, index)
        self.index = torch.tensor(index, dtype=torch.int32)
        # self.index, _ = torch.sort(self.index)

        # self.vectorized_has = torch.vmap(self.has)

        # if isinstance(data, torch.Tensor):
        #     self.data = torch.tensor(self.data, dtype=data.dtype)
        # elif isinstance(data, np.ndarray):
        #     self.data = np.array(self.data, dtype=data.dtype)

    def has(self, index):
        """
        Function to check if the partition has the given index

        Parameters
        ----------
        index : int

        Returns
        -------
        bool
            `True` if the partition has the given index, `False` otherwise

        """
        possible_index = torch.searchsorted(self.index, index)

        return index < len(self.index) and self.index[possible_index] == index

    def __len__(self):
        """
        Function to retrieve the length

        Returns
        -------
        int
            Number of items in the data

        """
        return len(self.index)

    def __getitem__(self, index):
        """
        Retrieves the item in data with the given index

        Parameters
        ----------
        index : int

        Returns
        -------
        Data
            The data sample with the given `index` in the dataset

        """
        return self.data[index]

    def get_trainset(self, batch_size=1, shuffle=True):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

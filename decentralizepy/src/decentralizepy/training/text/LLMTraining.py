import logging

import torch
from tqdm.auto import tqdm

from decentralizepy import utils
from decentralizepy.training.Training import Training


class LLMTraining(Training):
    """
    This class implements the training module for a single node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss=None,
        log_dir=".",
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
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
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            model,
            optimizer,
            loss,
            log_dir,
            rounds,
            full_epochs,
            batch_size,
            shuffle,
        )

    def eval_loss(self, dataset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in trainset:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs[0]
                epoch_loss += loss * len(input_ids)
                count += len(input_ids)
        loss = (epoch_loss / count).item()
        logging.info("Loss after iteration: {}".format(loss))
        self.model = self.model.cpu()
        return loss

    def trainstep(self, batch):
        """
        One training step on a minibatch.

        Parameters
        ----------
        batch : any
            Data item

        Returns
        -------
        int
            Loss Value for the step

        """
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        self.optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)

        for epoch in range(self.rounds):
            # print("Epoch: ", epoch)
            # progress_bar = tqdm(range(len(trainset)))
            epoch_loss = 0.0
            count = 0
            if self.rank == 0:
                for (
                    batch
                ) in (
                    trainset
                ):  # tqdm(trainset, desc=f"Epoch {epoch + 1}/{self.rounds}", leave=False):
                    logging.debug(
                        "Starting minibatch {} with num_samples: {}".format(
                            count, len(batch["input_ids"])
                        )
                    )
                    epoch_loss += self.trainstep(batch)
                    count += 1
            else:
                for batch in trainset:
                    logging.debug(
                        "Starting minibatch {} with num_samples: {}".format(
                            count, len(batch["input_ids"])
                        )
                    )
                    epoch_loss += self.trainstep(batch)
                    count += 1
                # progress_bar.update(1)
            # logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
            # print("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
            logging.info("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
            print("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
            # progress_bar.refresh()
            # progress_bar.reset()

    def train(self, dataset):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.train()
        if torch.cuda.is_available():
            self.model.cuda()

        if self.full_epochs:
            self.train_full(dataset)
        else:
            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            while count < self.rounds:
                for data in trainset:
                    iter_loss += self.trainstep(data)
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break

        self.model = self.model.cpu()

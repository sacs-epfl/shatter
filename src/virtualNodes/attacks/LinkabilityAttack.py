import logging

import torch


class LinkabilityAttack:
    """
    Class for mounting linkability attack on models in Collaborative Learning.

    """

    def __init__(self, num_clients, client_trainsets, loss) -> None:
        self.num_clients = num_clients
        self.client_trainsets = client_trainsets
        self.loss = loss

    def eval_loss(self, model, trainset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate
        trainset : torch.utils.data.DataLoader or decentralizepy.datasets.Data
            Training set to evaluate on

        """
        if torch.cuda.is_available():
            model = model.cuda()
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                output = model(data)
                loss_val = self.loss(output, target)
                epoch_loss = loss_val * len(target) + epoch_loss
                count += len(target)
            loss = epoch_loss / count
            loss = loss.item()
            logging.debug("Loss after iteration: {}".format(loss))
            return loss

    def attack(self, model, skip=[]):
        """
        Function to mount linkability attack on the model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be attacked.

        Returns
        -------
        int
            Dataset ID which is the most likely to be the dataset used to train the model.

        """
        with torch.no_grad():
            min_loss = 10e10
            predicted_client = None
            for client in self.client_trainsets:
                if client not in skip:
                    cur_loss = self.eval_loss(model, self.client_trainsets[client])
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        predicted_client = client
            return predicted_client

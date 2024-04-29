import logging

import torch

from virtualNodes.attacks.LinkabilityAttack import LinkabilityAttack


class LinkabilityAttackTwitter(LinkabilityAttack):
    """
    Class for mounting linkability attack on models in Collaborative Learning.

    """

    def __init__(self, num_clients, client_trainsets, *args, **kwargs) -> None:
        super().__init__(num_clients, client_trainsets, loss=None)

    def eval_loss(self, model, trainset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        if torch.cuda.is_available():
            model = model.cuda()
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in trainset:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                epoch_loss += loss * len(input_ids)
                count += len(input_ids)
        loss = (epoch_loss / count).item()
        # logging.info("Loss after iteration: {}".format(loss))
        model = model.cpu()
        return loss

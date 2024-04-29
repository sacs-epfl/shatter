import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.stats import norm
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from decentralizepy.datasets.CIFAR10 import ResNet18
from decentralizepy.training.Training import Training
from virtualNodes.attacks.MIA.LiRAPartitioner import LiRAPartitioner

NUM_CLASSES = 10


class LOSSMIA:
    def __init__(self):
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )
        # self.trainset = torchvision.datasets.CIFAR10(
        #     root=train_dir, train=True, download=True, transform=self.transform
        # )
        # self.dataset_size = len(self.trainset)
        # self.dataloader = DataLoader(self.trainset, batch_size=test_batch_size, shuffle=False)
        # print("Partitions Loaded...")
        # print("Initialized the new version of LOSSMIA")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def model_eval(self, model, data_samples, epsilon=10e-9):
        with torch.no_grad():
            model = model.to(self.device)
            data, targets = data_samples
            data = data.to(self.device)
            targets = targets.to(self.device)
            output = model(data)
            loss_val = (
                F.cross_entropy(output, targets, reduction="none").detach().clone()
            )

            nan_mask = torch.isnan(loss_val)
            loss_val[nan_mask] = torch.tensor(1 / epsilon).to(self.device)
            inf_mask = torch.isinf(loss_val)
            loss_val[inf_mask] = torch.tensor(1 / epsilon).to(self.device)

            return loss_val

    def attack_dataset(
        self,
        victim_model,
        in_dataloader,
        out_dataloader,
        in_size=10000,
        out_size=10000,
        epsilon=10e-9,
    ):
        victim_model.eval()
        loss_vals = {
            "in": torch.zeros((in_size,), dtype=torch.float32, device=self.device),
            "out": torch.zeros((out_size,), dtype=torch.float32, device=self.device),
        }
        with torch.no_grad():
            last = 0
            for data_samples in in_dataloader:
                loss_in = -self.model_eval(victim_model, data_samples, epsilon=epsilon)
                loss_vals["in"][last : last + len(data_samples[1])] = loss_in
                last += len(data_samples[1])
            loss_vals["in"] = loss_vals["in"][:last].cpu()

            last = 0
            for data_samples in out_dataloader:
                loss_out = -self.model_eval(victim_model, data_samples, epsilon=epsilon)
                loss_vals["out"][last : last + len(data_samples[1])] = loss_out
                last += len(data_samples[1])
            loss_vals["out"] = loss_vals["out"][:last].cpu()
            return loss_vals

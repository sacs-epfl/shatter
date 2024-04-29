import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.datasets.text.LLMData import LLMData
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model


def __read_file__(file_path):
    """
    Read data from the given json file

    Parameters
    ----------
    file_path : str
        The file path

    Returns
    -------
    tuple
        (users, num_samples, data)

    """
    with open(file_path, "r") as inf:
        client_data = json.load(inf)
    return (
        client_data["users"],
        client_data["num_samples"],
        client_data["user_data"],
    )


train_dir = "/nfs/risharma/sent140/per_user_data/train"
num_partitions = 100
sizes = None
dataset_id = 0
at_most = 1
NUM_CLASSES = 2

files = os.listdir(train_dir)
files = [f for f in files if f.endswith(".json")]
files.sort()
c_len = len(files)

# rng = Random()
# rng.seed(self.random_seed)
# rng.shuffle(files)

if sizes == None:  # Equal distribution of data among processes
    e = c_len // num_partitions
    frac = e / c_len
    sizes = [frac] * num_partitions
    sizes[-1] += 1.0 - frac * num_partitions

print("Dataset ID: ", dataset_id)

my_clients_temp = DataPartitioner(files, sizes).use(dataset_id)

my_clients = []
for i, x in enumerate(my_clients_temp):
    my_clients.append(x)
    i += 1
    if at_most and i >= at_most:
        break

print(my_clients)
my_train_data = {"x": [], "y": []}
clients = []
num_samples = []
for i in range(my_clients.__len__()):
    cur_file = my_clients.__getitem__(i)

    clients, _, train_data = __read_file__(os.path.join(train_dir, cur_file))
    for cur_client in clients:
        my_train_data["x"].extend([x[4] for x in train_data[cur_client]["x"]])
        my_train_data["y"].extend(
            [0 if x == "0" else 1 for x in train_data[cur_client]["y"]]
        )
        num_samples.append(len(train_data[cur_client]["y"]))

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)

train_y = torch.nn.functional.one_hot(
    torch.tensor(my_train_data["y"]).to(torch.int64),
    num_classes=NUM_CLASSES,
).to(torch.float32)
train_x = tokenizer(
    my_train_data["x"], return_tensors="pt", truncation=True, padding=True
)

print("train_x: ", train_x)
print("train_y: ", train_y)

import json
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.distributions import Normal
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from decentralizepy.datasets.text.LLMData import LLMData
from decentralizepy.datasets.text.Twitter import BERT, Twitter
from decentralizepy.mappings.Linear import Linear
from decentralizepy.training.text.LLMTraining import LLMTraining
from virtualNodes.attacks.MIA.LiRAPartitioner import LiRAPartitioner

NUM_CLASSES = 2


class TwitterDataset:
    def __init__(self, tup):
        self.tup = tup

    def get_trainset(self, batch_size=1, shuffle=True):
        return DataLoader(
            LLMData(self.tup[0], self.tup[1]), batch_size=batch_size, shuffle=shuffle
        )


class LiRATwitter:

    def __read_file__(self, file_path):
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

    def dataloader_from_file(self, idx):
        files = self.training_partitions.use(idx)
        my_train_data = {"x": [], "y": []}

        for file in files:
            clients, _, train_data = self.__read_file__(
                os.path.join(self.train_dir, file)
            )
            for cur_client in clients:
                my_train_data["x"].extend([x[4] for x in train_data[cur_client]["x"]])
                my_train_data["y"].extend(
                    [0 if x == "0" else 1 for x in train_data[cur_client]["y"]]
                )

        train_y = torch.nn.functional.one_hot(
            torch.tensor(my_train_data["y"]).to(torch.int64),
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
        train_x = self.tokenizer(
            my_train_data["x"], return_tensors="pt", truncation=True, padding=True
        )
        assert train_x["input_ids"].shape[0] == train_y.shape[0]
        assert train_y.shape[0] > 0

        return train_x, train_y, train_y.shape[0]

    def __init__(
        self,
        K=64,
        random_seed=34252,
        train_dir="/train",
    ):
        self.K = K
        self.random_seed = random_seed
        self.train_dir = train_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", model_max_length=512
        )

        files = os.listdir(train_dir)
        files.sort()

        print("Partitioning for shadow models.")
        self.training_partitions = LiRAPartitioner(
            files, K=self.K, seed=self.random_seed
        )

        # self.training_sets = []
        # self.num_samples = []
        # self.dataset_size = 0
        # for i in range(self.K):
        #     dl = self.dataloader_from_file(i)
        #     self.num_samples.append(dl[2])
        #     self.dataset_size += dl[2]
        #     self.training_sets.append(TwitterDataset(dl[:2]))

    def train_batch_shadow_models(
        self, start, count, epochs, weights_store_dir="./weights", is_parallel=False
    ):
        if is_parallel:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        for i in range(start, min(start + count, self.K)):
            print("Training shadow model ", i)
            my_model = BERT()
            optimizer = SGD(my_model.parameters(), lr=0.005)
            training = LLMTraining(
                0,
                0,
                None,
                my_model,
                optimizer,
                None,
                "./",
                rounds=epochs,
                full_epochs=True,
                batch_size=64,
                shuffle=True,
            )
            print("Loaded training ", i)
            dl = TwitterDataset(self.dataloader_from_file(i)[:2])
            print("Training partition loading complete.")
            training.train(dl)
            print("Completed training, saving model ", i)
            my_model.save_pretrained(
                os.path.join(weights_store_dir, "shadow_{}".format(i)), from_pt=True
            )

    def train_parallel_shadow_models(self, epochs, weights_store_dir="./weights"):
        import math
        import os
        from multiprocessing import Pool

        number_of_processes = 4  # os.cpu_count()
        pool = Pool(processes=number_of_processes)
        for i in range(number_of_processes):
            start = math.floor(i * self.K / number_of_processes)
            count = math.floor(self.K / number_of_processes)
            pool.apply_async(
                self.train_batch_shadow_models,
                args=(start, count, epochs, weights_store_dir, True),
            )
        pool.close()
        pool.join()

    # def load_shadow_models(self, weights_store_dir):
    #     shadow_model_dict = dict()
    #     for i in range(self.K):
    #         shadow_model_dict[i] = BERT(path = os.path.join(weights_store_dir,"shadow_{}".format(i)))
    #         # shadow_model_dict[i].load_state_dict(torch.load(os.path.join(weights_store_dir,"shadow_{}.pt".format(i))))
    #     return shadow_model_dict


class LiRAMIA:
    def __init__(self, shadow_dataset_model: LiRATwitter, weights_store_dir):
        self.random_seed = 1234
        self.weights_store_dir = weights_store_dir
        self.shadow_dataset_model = shadow_dataset_model
        # self.shadow_model_dict = shadow_dataset_model.load_shadow_models(weights_store_dir=weights_store_dir)
        print("Creating and Loading partitions")
        self.trainset = Twitter(
            0,
            0,
            Linear(1, 1),
            self.random_seed,
            only_local=False,
            train_dir=self.shadow_dataset_model.train_dir,
            test_batch_size=64,
            tokenizer="BERT",
            at_most=None,
        )
        # self.shadow_training_partitions = {x : self.shadow_dataset_model.training_partitions.use(x) for x in range(self.shadow_dataset_model.K)}
        print("Partitions Loaded...")
        self.num_classes = NUM_CLASSES
        self.confidences = torch.zeros(
            (self.shadow_dataset_model.K, self.trainset.train_y.shape[0]),
            dtype=torch.float32,
        )
        if torch.cuda.is_available():
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.confidences = self.confidences.to(self.device)

        self.index_ranges = dict()
        start_index = 0
        for client, num_samples in zip(
            self.trainset.clients, self.trainset.num_samples
        ):
            self.index_ranges[client] = (start_index, start_index + num_samples)
            start_index += num_samples

    def model_confidence(self, model, data_samples, epsilon=10e-9):
        with torch.no_grad():
            model = model.to(self.device)
            data_samples = {k: v.to(self.device) for k, v in data_samples.items()}
            # data, targets = data_samples
            input_ids = data_samples["input_ids"]
            attention_mask = data_samples["attention_mask"]
            targets = data_samples["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            logits = outputs.logits
            loss_val = (
                F.cross_entropy(logits, targets, reduction="none").detach().clone()
            )

            p = F.softmax(logits, dim=-1)

            # For each row (0th dimension) in p, take the value at the index of the target
            p = p[range(p.shape[0]), torch.argmax(targets, dim=1)]

            p = torch.clamp(p, epsilon, 1 - epsilon)  # stabilization

            ratio = torch.clamp(p / (1 - p), epsilon, 1 / epsilon)

            # model = model.cpu()
            return torch.log(ratio), loss_val

    def load_shadow_confidences(self, confidence_file_path):
        self.confidences = torch.load(confidence_file_path).to(self.device)
        # if torch.cuda.is_available():
        #     self.confidences = self.confidences.cuda()
        # self.confidences = self.confidences.cpu()

    def precompute_shadow_confidences(self, batch_size=128):
        dataloader = self.trainset.get_trainset(batch_size=batch_size, shuffle=False)

        for shadow_id in range(self.shadow_dataset_model.K):
            shadow_model = BERT(
                path=os.path.join(self.weights_store_dir, "shadow_{}".format(shadow_id))
            )
            last = 0
            print("Precomputing confidences for shadow model ", shadow_id)
            shadow_model.eval()
            for data_samples in tqdm(dataloader, leave=True):
                confidences, _ = self.model_confidence(shadow_model, data_samples)
                cur_batch_len = confidences.shape[0]
                self.confidences[shadow_id, last : last + cur_batch_len] = confidences
                last += cur_batch_len
            torch.save(
                self.confidences,
                os.path.join(
                    self.weights_store_dir,
                    "confidences_clamped_seed{}_K{}.pt".format(
                        self.random_seed, self.shadow_dataset_model.K
                    ),
                ),
            )
        self.confidences = self.confidences.cpu()
        torch.save(
            self.confidences,
            os.path.join(
                self.weights_store_dir,
                "confidences_clamped_seed{}_K{}.pt".format(
                    self.random_seed, self.shadow_dataset_model.K
                ),
            ),
        )

    # def attack_batch(self, victim_model, data_indices : torch.Tensor, data_samples : torch.Tensor, epsilon = 10e-9):
    #     batch_size = len(data_indices)
    #     confs_out = [[] for _ in range(batch_size)]
    #     confs_in = [[] for _ in range(batch_size)]

    #     for shadow_id in range(self.shadow_dataset_model.K):
    #         my_clients = list(map(lambda x: x[:-5] , list(self.shadow_dataset_model.training_partitions.use(shadow_id))))
    #         # print("my_clients: ", my_clients)
    #         in_partition = torch.zeros(data_indices.numel(), dtype = torch.bool)
    #         for i, index in enumerate(data_indices):
    #             for client in my_clients:
    #                 if self.index_ranges[client][0] <= index and index < self.index_ranges[client][1]:
    #                     in_partition[i] = True

    #     # for shadow_id in range(self.shadow_dataset_model.K):
    #     #     my_clients = [x[:-5] for x in self.shadow_dataset_model.training_partitions.use(shadow_id)]
    #     #     in_partition = torch.zeros(data_indices.numel(), dtype=torch.bool, device=data_indices.device)

    #     #     for client in my_clients:
    #     #         client_start, client_end = self.index_ranges[client]
    #     #         # Vectorized comparison
    #     #         in_partition |= (data_indices >= client_start) & (data_indices < client_end)

    #         selected_confs = self.confidences[shadow_id, data_indices]

    #         for idx, (is_in_partition, conf) in enumerate(zip(in_partition, selected_confs)):
    #             if not is_in_partition:
    #                 confs_out[idx].append(conf)
    #             else:
    #                 confs_in[idx].append(conf)
    #         # print("data_indices", data_indices)
    #         # break

    #     confs_out = [torch.tensor(confs) for confs in confs_out]
    #     # print("confs_out: ", confs_out)
    #     mean_out = torch.tensor([torch.mean(confs, dim=0) for confs in confs_out])
    #     std_out = torch.clamp(torch.tensor([torch.std(confs, dim=0) for confs in confs_out]), epsilon, 1/epsilon)
    #     N_out = Normal(mean_out, std_out)

    #     victim_confs, loss_val = self.model_confidence(victim_model, data_samples)
    #     victim_confs = victim_confs.cpu()
    #     loss_val = loss_val.cpu()
    #     # print("victim_confs: ", victim_confs)

    #     confs_in = [torch.tensor(confs) for confs in confs_in]
    #     # print("confs_in: ", confs_in)
    #     mean_in = torch.tensor([torch.mean(confs, dim=0) for confs in confs_in])
    #     # print("mean_in: ", mean_in)
    #     std_in = torch.clamp(torch.tensor([torch.std(confs, dim=0) for confs in confs_in]), epsilon, 1/epsilon)
    #     # print("std_in: ", std_in)
    #     N_in = Normal(mean_in, std_in)
    #     self.N_in = N_in
    #     self.N_out = N_out
    #     p_in = torch.clamp(N_in.log_prob(victim_confs).exp(), epsilon, 1/epsilon)
    #     p_out = torch.clamp(N_out.log_prob(victim_confs).exp(), epsilon, 1/epsilon)
    #     ret_online = torch.clamp(p_in / p_out, epsilon, 1/epsilon)
    #     return N_out.cdf(victim_confs), ret_online, 1 - loss_val

    def attack_batch(
        self,
        victim_model,
        data_indices: torch.Tensor,
        data_samples: torch.Tensor,
        epsilon=10e-9,
    ):
        batch_size = len(data_indices)
        data_indices = data_indices.to(self.device)
        confs_out = torch.zeros(
            (batch_size, self.shadow_dataset_model.K // 2), device=self.device
        )
        confs_in = torch.zeros(
            (batch_size, self.shadow_dataset_model.K // 2), device=self.device
        )
        in_indices = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        out_indices = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)

        for shadow_id in range(self.shadow_dataset_model.K):
            my_clients = list(
                map(
                    lambda x: x[:-5],
                    list(self.shadow_dataset_model.training_partitions.use(shadow_id)),
                )
            )
            # print("my_clients: ", len(my_clients))
            index_ranges = torch.tensor(
                [self.index_ranges[client] for client in my_clients], device=self.device
            )

            in_partition = (
                (data_indices[:, None] >= index_ranges[:, 0])
                & (data_indices[:, None] < index_ranges[:, 1])
            ).any(dim=1)

            # print("in_partition: ", in_partition.shape, in_partition)
            not_in_partition = ~in_partition
            selected_confs = self.confidences[shadow_id, data_indices].to(self.device)
            # print(selected_confs.shape, selected_confs[not_in_partition].shape, selected_confs[in_partition].shape)
            rows_to_update_out = torch.where(not_in_partition)[0]
            rows_to_update_in = torch.where(in_partition)[0]
            # print("confs_out[not_in_partition, out_indices]", confs_out[rows_to_update_out, out_indices[rows_to_update_out]])
            confs_out[rows_to_update_out, out_indices[rows_to_update_out]] = (
                selected_confs[not_in_partition]
            )
            confs_in[rows_to_update_in, in_indices[rows_to_update_in]] = selected_confs[
                in_partition
            ]
            out_indices[rows_to_update_out] += 1
            in_indices[rows_to_update_in] += 1

            # confs_out = torch.cat((confs_out, selected_confs[~in_partition].t()[None, :]), dim=0)
            # confs_in = torch.cat((confs_in, selected_confs[in_partition].t()[None, :]), dim=0)

        # confs_out = confs_out.t()
        # confs_in = confs_in.t()
        # print(confs_out.shape, confs_in.shape)

        # # Ensuring the output shapes are (-1, batch_size)
        # confs_out = confs_out.t()
        # confs_in = confs_in.t()

        # return confs_out, confs_in

        # confs_out = [torch.tensor(confs) for confs in confs_out]
        # print("confs_out: ", confs_out)
        mean_out = torch.mean(confs_out, dim=1)
        std_out = torch.clamp(torch.std(confs_out, dim=1), epsilon, 1 / epsilon)
        N_out = Normal(mean_out, std_out)

        victim_confs, loss_val = self.model_confidence(victim_model, data_samples)
        # victim_confs = victim_confs.cpu()
        # loss_val = loss_val.cpu()
        # print("victim_confs: ", victim_confs)

        # confs_in = [torch.tensor(confs) for confs in confs_in]
        # print("confs_in: ", confs_in)
        mean_in = torch.mean(confs_in, dim=1)
        # print("mean_in: ", mean_in)
        std_in = torch.clamp(torch.std(confs_in, dim=1), epsilon, 1 / epsilon)
        # print("std_in: ", std_in)
        N_in = Normal(mean_in, std_in)
        # self.N_in = N_in
        # self.N_out = N_out
        p_in = torch.clamp(N_in.log_prob(victim_confs).exp(), epsilon, 1 / epsilon)
        p_out = torch.clamp(N_out.log_prob(victim_confs).exp(), epsilon, 1 / epsilon)
        ret_online = torch.clamp(p_in / p_out, epsilon, 1 / epsilon)
        return N_out.cdf(victim_confs), ret_online, 1 - loss_val

    def attack_dataset(
        self,
        victim_model,
        batch_size=512,
        online=True,
        epsilon=10e-9,
        return_loss=False,
        return_both=False,
    ):
        dataloader = self.trainset.get_trainset(batch_size=batch_size, shuffle=False)
        dataset_size = self.confidences.shape[1]
        victim_model.eval()
        likelihood_ratios_online = (
            torch.zeros((dataset_size,), dtype=torch.float32, device=self.device)
            if online
            else None
        )
        likelihood_ratios_offline = (
            torch.zeros((dataset_size,), dtype=torch.float32, device=self.device)
            if return_both or not online
            else None
        )
        loss_vals = (
            torch.zeros((dataset_size,), dtype=torch.float32, device=self.device)
            if return_loss
            else None
        )

        last = 0
        with torch.no_grad():
            for data_samples in dataloader:
                # print("Attacking with batch {}".format(last))
                index = torch.arange(last, last + len(data_samples["labels"]))
                last += len(data_samples["labels"])
                # print("last: ", last)
                lir_offline, lir_online, loss = self.attack_batch(
                    victim_model, index, data_samples, epsilon=epsilon
                )
                if likelihood_ratios_online is not None:
                    likelihood_ratios_online[index] = lir_online
                if likelihood_ratios_offline is not None:
                    likelihood_ratios_offline[index] = lir_offline
                if loss_vals is not None:
                    loss_vals[index] = loss
            # self.attack_batch(victim_model, index, data_samples, data_targets, online = online)
            if likelihood_ratios_offline is not None:
                likelihood_ratios_offline = likelihood_ratios_offline.cpu()
            if likelihood_ratios_online is not None:
                likelihood_ratios_online = likelihood_ratios_online.cpu()
            if loss_vals is not None:
                loss_vals = loss_vals.cpu()
        return likelihood_ratios_offline, likelihood_ratios_online, loss_vals


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset = LiRATwitter(K=64, random_seed=34252, train_dir="/train")
    dataset.train_parallel_shadow_models(
        epochs=20, weights_store_dir="./weights_BERT_64_Full"
    )
    mia = LiRAMIA(dataset, "./weights_BERT_64_Full").precompute_shadow_confidences(
        batch_size=512
    )

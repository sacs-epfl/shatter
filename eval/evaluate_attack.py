import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import sys

def process_directory(directory_path):
    attack_dicts_algorithms = {}

    algorithms = sorted(os.listdir(directory_path))

    for algorithm in algorithms:
        algorithm_path = os.path.join(directory_path, algorithm)
        print("Processing Path: ", algorithm_path)
        if os.path.isdir(algorithm_path):
            attack_dicts_algorithms[algorithm] = {}
            machine_folders = os.listdir(algorithm_path)
            machine_folders = [machine_folder for machine_folder in machine_folders if machine_folder.startswith("machine")]
            for machine in machine_folders:
                machine_path = os.path.join(algorithm_path, machine)
                print("Processing machine path: ", machine_path)
                if os.path.isdir(machine_path):
                    for filename in os.listdir(machine_path):
                        if filename.endswith("_attacker.pth"):
                            torch_file_path = os.path.join(machine_path, filename)

                            try:
                                attacker_File = torch.load(torch_file_path)
                                for attack in attacker_File.keys():
                                    if attack not in attack_dicts_algorithms[algorithm]:
                                        attack_dicts_algorithms[algorithm][attack] = {}
                                    for victim_client in attacker_File[attack].keys():
                                        if victim_client not in attack_dicts_algorithms[algorithm][attack]:
                                            attack_dicts_algorithms[algorithm][attack][victim_client] = {}
                                        for iteration in attacker_File[attack][victim_client].keys():
                                            if iteration not in attack_dicts_algorithms[algorithm][attack][victim_client]:
                                                attack_dicts_algorithms[algorithm][attack][victim_client][iteration] = []
                                            attack_dicts_algorithms[algorithm][attack][victim_client][iteration].extend(attacker_File[attack][victim_client][iteration])

                            except FileNotFoundError:
                                print(f"File not found: {torch_file_path}")

                            except json.JSONDecodeError:
                                print(f"Error decoding JSON in the file: {torch_file_path}")
    return attack_dicts_algorithms

def get_roc_auc(a):
    num_true = a["in"].numel()
    num_false = a["out"].numel()

    print("Number of non-training samples: ", num_false)

    y_true_balanced = torch.zeros((num_true + num_false,), dtype = torch.int32)
    y_true_balanced[:num_true] = 1

    y_pred = a
    y_pred_balanced = torch.zeros((num_true + num_false,), dtype=torch.float32)
    y_pred_balanced[:num_true] = a["in"]
    y_pred_balanced[num_true:] = y_pred["out"]

    # # Use the balanced y_pred for the ROC curve
    y_pred = y_pred_balanced.numpy()
    y_true = y_true_balanced.numpy()
    print("Shapes: ", y_pred.shape, y_true.shape)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return fpr, tpr, thresholds, roc_auc

def get_auc_means_clients(attack_dict):
    auc_means_clients = []
    auc_stdev_clients = []
    counts = []
    for victim_client in attack_dict:
        aucs = []
        iterations_to_attack = sorted(list(attack_dict[victim_client].keys()))
        for iteration in iterations_to_attack:
            if iteration in attack_dict[victim_client].keys():
                for attack_by_each_attacker in attack_dict[victim_client][iteration]:
                    _,_,_,roc_auc = get_roc_auc(attack_by_each_attacker)
                    aucs.append(roc_auc)

        aucs = torch.tensor(aucs)
        auc_means_clients.append(torch.mean(aucs))
        auc_stdev_clients.append(torch.std(aucs))
        counts.append(aucs.numel())
    auc_means_clients, sort_indices = torch.sort(torch.tensor(auc_means_clients), descending = False)
    auc_stdev_clients = torch.tensor(auc_stdev_clients)[sort_indices]
    counts = torch.tensor(counts)[sort_indices]
    return auc_means_clients, auc_stdev_clients, counts

def get_auc_means_iterations(attack_dict, iterations_to_attack):
    auc_means_iterations = []
    auc_stdev_iterations = []
    counts = []
    for iteration in iterations_to_attack:
        aucs = []
        for victim_client in attack_dict:
            if iteration in attack_dict[victim_client].keys():
                for attack_by_each_attacker in attack_dict[victim_client][iteration]:
                    _, _, _, roc_auc = get_roc_auc(attack_by_each_attacker)
                    aucs.append(roc_auc)

        aucs = torch.tensor(aucs)
        auc_means_iterations.append(torch.mean(aucs))
        auc_stdev_iterations.append(torch.std(aucs))
        counts.append(aucs.numel())

    auc_means_iterations = torch.tensor(auc_means_iterations)
    auc_stdev_iterations = torch.tensor(auc_stdev_iterations)
    counts = torch.tensor(counts)
    return auc_means_iterations, auc_stdev_iterations, counts

def get_linkability_means_clients(attack_dict):
    linkability_means_clients = []
    linkability_stdev_clients = []
    counts = []
    for victim_client in attack_dict:
        linkabilities = []
        iterations_to_attack = sorted(list(attack_dict[victim_client].keys()))
        for iteration in iterations_to_attack:
            if iteration in attack_dict[victim_client].keys():
                correct_pred = 0
                total_pred = 0
                for attack_by_each_attacker in attack_dict[victim_client][iteration]:
                    if attack_by_each_attacker == victim_client:
                        correct_pred += 1
                    total_pred += 1
                linkabilities.append(correct_pred/total_pred)
        linkabilities = torch.tensor(linkabilities)
        linkability_means_clients.append(torch.mean(linkabilities))
        linkability_stdev_clients.append(torch.std(linkabilities))
        counts.append(linkabilities.shape[0])
    linkability_means_clients, sort_indices = torch.sort(torch.tensor(linkability_means_clients), descending = False)
    linkability_stdev_clients = torch.tensor(linkability_stdev_clients)[sort_indices]
    counts = torch.tensor(counts)[sort_indices]
    return linkability_means_clients, linkability_stdev_clients, counts


def get_linkability_means_iterations(attack_dict, iterations_to_attack):
    linkability_means_clients = []
    linkability_stdev_clients = []
    counts = []
    for iteration in iterations_to_attack:
        linkabilities = []
        for victim_client in attack_dict:
            if iteration in attack_dict[victim_client].keys():
                correct_pred = 0
                total_pred = 0
                for attack_by_each_attacker in attack_dict[victim_client][iteration]:
                    if attack_by_each_attacker == victim_client:
                        correct_pred += 1
                    total_pred += 1
                linkabilities.append(correct_pred/total_pred)
        linkabilities = torch.tensor(linkabilities)
        linkability_means_clients.append(torch.mean(linkabilities))
        linkability_stdev_clients.append(torch.std(linkabilities))
        counts.append(linkabilities.shape[0])
    linkability_means_clients, sort_indices = torch.sort(torch.tensor(linkability_means_clients), descending = False)
    linkability_stdev_clients = torch.tensor(linkability_stdev_clients)[sort_indices]
    counts = torch.tensor(counts)
    return linkability_means_clients, linkability_stdev_clients, counts



if __name__ == "__main__":

    # Check if there are 2 arguments
    if len(sys.argv) != 2:
        print("Usage: python evaluate_attack.py <root_directory of the dataset>")
        sys.exit(1)
    root_directory = sys.argv[1]

    random_seed = 90
    attack_dict_vnodes = process_directory(root_directory)

    # MIA Attack
    print("Processing MIA Attack")
    print("---------------------")

    evaluating_client = 0
    attack_dicts = {
        key: value["loss_vals"] for key, value in attack_dict_vnodes.items()
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label in attack_dicts:
        if 'iterations' in attack_dicts[label][evaluating_client].keys():
            iterations_to_attack = sorted(list(attack_dicts[label][evaluating_client]['iterations'].keys()))
        else:
            iterations_to_attack = sorted(list(attack_dicts[label][evaluating_client].keys()))
            clients = torch.arange(len(list(attack_dicts[label].keys()))) + 1
        auc_means_clients, auc_stdev_clients, counts_clients = get_auc_means_clients(attack_dicts[label])
        auc_means_iterations, auc_stdev_iterations, counts_iterations = get_auc_means_iterations(attack_dicts[label], iterations_to_attack)

        # Write the Clients AUCs to a csv file. Use pandas to write to csv file. Also include a column for counts_clients
        df = pd.DataFrame({'Client ID': clients, 'mean': auc_means_clients.numpy(), 'std': auc_stdev_clients.numpy(), 'nr_iters': counts_clients.numpy()})
        df.to_csv(f'{root_directory}/{label}_clients_MIA.csv', index=False)
        df = pd.DataFrame({'Iterations': iterations_to_attack, 'mean': auc_means_iterations.numpy(), 'std': auc_stdev_iterations.numpy(), 'nr_nodes': counts_iterations.numpy()})
        df.to_csv(f'{root_directory}/{label}_iterations_MIA.csv', index = False)

        ax1.plot(clients, auc_means_clients, label=label)
        ax1.fill_between(clients, auc_means_clients - auc_stdev_clients, auc_means_clients + auc_stdev_clients, alpha=0.1, lw=2)
        ax2.plot(iterations_to_attack, auc_means_iterations, label=label)
        ax2.fill_between(iterations_to_attack, auc_means_iterations - auc_stdev_iterations, auc_means_iterations + auc_stdev_iterations, alpha=0.1, lw=2)

    ax1.set_title('MIA per Client')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('AUC')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.5,1.0)

    ax2.set_title('MIA per Iteration')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('AUC')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="lower right")
    ax2.set_ylim(0.5,1.0)


    fig.savefig(f'{root_directory}/VNodesMIA.pdf', dpi=300, bbox_inches='tight')



    # Linkability Attack
    print()
    print()
    print("Processing Linkability Attack")
    print("---------------------")



    attack_dicts = {
        key: value["linkability"] for key, value in attack_dict_vnodes.items()
    }

    evaluating_client = 0


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label in attack_dicts:
        iterations_to_attack = sorted(list(attack_dicts[label][evaluating_client].keys()))
        if 'iterations' in attack_dicts[label][evaluating_client].keys():
            iterations_to_attack = sorted(list(attack_dicts[label][evaluating_client]['iterations'].keys()))
            clients = torch.arange(len(list(attack_dicts[label].keys()))) + 1
        else:
            iterations_to_attack = sorted(list(attack_dicts[label][evaluating_client].keys()))
        mean_clients, std_clients, counts_clients = get_linkability_means_clients(attack_dicts[label])
        mean_iterations, std_iterations, counts_iterations = get_linkability_means_iterations(attack_dicts[label], iterations_to_attack)

        # Write the Clients AUCs to a csv file. Use pandas to write to csv file. Also include a column for counts_clients
        df = pd.DataFrame({'Client ID': clients, 'mean': mean_clients.numpy(), 'std': std_clients.numpy(), 'nr_iters': counts_clients.numpy()})
        df.to_csv(f'{root_directory}/{label}_clients_linkability.csv', index=False)
        df = pd.DataFrame({'Iterations': iterations_to_attack, 'mean': mean_iterations.numpy(), 'std': std_iterations.numpy(), 'nr_nodes': counts_iterations.numpy()})
        df.to_csv(f'{root_directory}/{label}_iterations_linkability.csv', index=False)

        ax1.plot(clients, mean_clients, label=label)
        ax1.fill_between(clients, mean_clients - std_clients, mean_clients + std_clients, alpha=0.1, lw=2)
        ax2.plot(iterations_to_attack, mean_iterations, label=label)
        ax2.fill_between(iterations_to_attack, mean_iterations - std_iterations, mean_iterations + std_iterations, alpha=0.1, lw=2)


    ax1.set_title('Linkability Attack per Client')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Linkability Attack Success (%)')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc="upper left")

    ax2.set_title('Linkability Attack per Iteration')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Linkability Attack Success (%)')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper left")


    fig.savefig(f'{root_directory}/VNodesLinkability.pdf', dpi=300, bbox_inches='tight')


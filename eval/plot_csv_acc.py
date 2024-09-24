import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MAX_ITERATION = None


def get_stats(l):
    assert len(l) > 0
    mean_dict, stdev_dict, min_dict, max_dict, counts_dict = {}, {}, {}, {}, {}
    for key in l[0].index:
        if MAX_ITERATION is not None and key >= MAX_ITERATION:
            continue
        all_nodes = [i[key] for i in l]
        all_nodes = np.array(all_nodes)
        mean = np.mean(all_nodes)
        std = np.std(all_nodes)
        min = np.min(all_nodes)
        max = np.max(all_nodes)
        count = np.count_nonzero(~np.isnan(all_nodes))
        mean_dict[int(key)] = mean
        stdev_dict[int(key)] = std
        min_dict[int(key)] = min
        max_dict[int(key)] = max
        counts_dict[int(key)] = count
    return mean_dict, stdev_dict, min_dict, max_dict, counts_dict


def plot(
    means,
    stdevs,
    mins,
    maxs,
    title,
    label,
    loc,
    xlabel="Training Epochs",
    ylabel="Top-1 Test Accuracy (%)",
):
    plt.title(title)
    plt.xlabel(xlabel)
    x_axis = np.array(list(means.keys()))
    y_axis = np.array(list(means.values()))
    err = np.array(list(stdevs.values()))
    plt.plot(x_axis, y_axis, label=label)
    plt.ylabel(ylabel)
    plt.fill_between(x_axis, y_axis - err, y_axis + err, alpha=0.4)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc=loc)


def replace_dict_key(d_org: dict, d_other: dict):
    result = {}
    for x, y in d_org.items():
        result[d_other[x]] = y
    return result


def create_list_of_metrics(results, metric):
    return [x[metric][x[metric].notna()] for x in results if metric in x]


def plot_results(path):
    folders = os.listdir(path)

    folders.sort()
    print("Reading folders from: ", path)
    print("Folders: ", folders)
    bytes_means, bytes_stdevs = {}, {}
    meta_means, meta_stdevs = {}, {}
    data_means, data_stdevs = {}, {}
    for folder in folders:
        folder_path = Path(os.path.join(path, folder))
        if not folder_path.is_dir() or "weights" == folder_path.name:
            continue
        results = []
        machine_folders = os.listdir(folder_path)
        for machine_folder in machine_folders:
            mf_path = os.path.join(folder_path, machine_folder)
            if not os.path.isdir(mf_path):
                continue
            files = os.listdir(mf_path)
            files = [f for f in files if f.endswith("_results.csv")]
            for f in files:
                filepath = os.path.join(mf_path, f)
                results.append(pd.read_csv(filepath, index_col=0))

        # Plot Training loss
        plt.figure(1)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "train_loss")
        )
        plot(means, stdevs, mins, maxs, "Training Loss", folder, "upper right")

        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )

        df.to_csv(os.path.join(path, f"{folder}_train_loss.csv"), index_label="rounds")
        # Plot Testing loss
        plt.figure(2)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "test_loss")
        )
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Convergence (Test Loss)",
            folder,
            "upper right",
            ylabel="Cross Entropy Loss",
        )
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(os.path.join(path, f"{folder}_test_loss.csv"), index_label="rounds")
        # Plot Testing Accuracy
        plt.figure(3)
        means, stdevs, mins, maxs, counts = get_stats(
            create_list_of_metrics(results, "test_acc")
        )
        plot(
            means,
            stdevs,
            mins,
            maxs,
            "Convergence (Test Accuracy)",
            folder,
            "lower right",
        )
        df = pd.DataFrame(
            {
                "mean": list(means.values()),
                "std": list(stdevs.values()),
                "nr_nodes": counts,
            },
            list(means.keys()),
            columns=["mean", "std", "nr_nodes"],
        )
        df.to_csv(os.path.join(path, f"{folder}_test_acc.csv"), index_label="rounds")

    plt.figure(1)
    plt.savefig(os.path.join(path, "train_loss.pdf"), dpi=600)
    plt.figure(2)
    plt.savefig(os.path.join(path, "test_loss.pdf"), dpi=600)
    plt.figure(3)
    plt.savefig(os.path.join(path, "test_acc.pdf"), dpi=600)


if __name__ == "__main__":
    assert len(sys.argv) == 2
    plot_results(sys.argv[1])

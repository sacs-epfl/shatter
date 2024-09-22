import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root = "./experiments/lenet"

results_dict = {}

baseline_folders = os.listdir(root)
# baseline_folders = ["EL", "TopK", "VNode2", "VNode4"]
for baseline_folder in baseline_folders:
    results_dict[baseline_folder] = {}
    baseline_path = os.path.join(root, baseline_folder)
    clients = os.listdir(baseline_path)
    for client in clients:
        client_path = os.path.join(baseline_path, client)
        if os.path.isdir(client_path):
            filePath = os.path.join(client_path, "fedavg.dat")
            print(filePath)
            assert os.path.exists(filePath)
            with open(filePath, "rb") as f:
                data = pickle.load(f)
                results_dict[baseline_folder][client] = data


# Assuming 'data' is your dictionary with all the details as described
data = results_dict

# Convert nested dictionary to DataFrame
rows = []
for baseline, experiments in data.items():
    for exp_id, metrics in experiments.items():
        for metric, value in metrics.items():
            rows.append({'Baseline': baseline, 'Metric': metric, 'Value': value})

df = pd.DataFrame(rows)

# Group by 'Baseline' and 'Metric', then calculate mean, std, and count (renamed to nr_nodes)
result_df = df.groupby(['Baseline', 'Metric']).agg(
    Mean=('Value', 'mean'),
    StdDev=('Value', 'std'),
    nr_nodes=('Value', 'count')
).reset_index()

# Create bar plots for each metric
for metric in result_df['Metric'].unique():
    metric_df = result_df[result_df['Metric'] == metric]
    metric_df.to_csv(f'{metric}_stats.csv', index=False)
    plt.figure(figsize=(10, 6))
    plt.bar(metric_df['Baseline'], metric_df['Mean'], yerr=metric_df['StdDev'], capsize=5)
    plt.title(f'Mean Values of {metric} by Baseline with Error Bars')
    plt.xlabel('Baseline')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{root}/{metric}_bar_plot.png')
    plt.close()
    print(f"Created {root}/{metric}_bar_plot.png")
    
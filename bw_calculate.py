import sys
import pandas as pd
import numpy as np
import itertools as it
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

world_size = int(sys.argv[1])
num_nodes = int(sys.argv[2])
bw_results_dir = sys.argv[3]

## We use the send data, since this consistently produces the min bandwidth


def times_intra_node(time_mat):
    ranks = list(range(world_size))
    prod = list(it.product(ranks, ranks))
    intra_pairs_x = []
    intra_pairs_y = []
    for pair in prod:
        if pair[0] != pair[1]:
            if (
                pair[0] < world_size / num_nodes and pair[1] < world_size / num_nodes
            ) or (
                pair[0] >= world_size / num_nodes and pair[1] >= world_size / num_nodes
            ):
                intra_pairs_x.append(pair[0])
                intra_pairs_y.append(pair[1])

    return time_mat[intra_pairs_x, intra_pairs_y]


def times_inter_node(time_mat):
    ranks = list(range(world_size))
    prod = list(it.product(ranks, ranks))
    inter_pairs_x = []
    inter_pairs_y = []
    for pair in prod:
        if pair[0] != pair[1]:
            if not (
                (pair[0] < world_size / num_nodes and pair[1] < world_size / num_nodes)
                or (
                    pair[0] >= world_size / num_nodes
                    and pair[1] >= world_size / num_nodes
                )
            ):
                inter_pairs_x.append(pair[0])
                inter_pairs_y.append(pair[1])

    return time_mat[inter_pairs_x, inter_pairs_y]


# Read send times
df = pd.read_csv(f"{bw_results_dir}/bw_0_send.data", header=None).rename(
    columns={0: "data_size"}
)
df["source_rank"] = 0
df = df.sort_values(by=["data_size", "source_rank"])
for rank in range(1, world_size):
    df2 = pd.read_csv(f"{bw_results_dir}/bw_{rank}_send.data", header=None).rename(
        columns={0: "data_size"}
    )
    df2["source_rank"] = rank
    df = pd.concat([df, df2])
    df = df.sort_values(by=["data_size", "source_rank"])

df = df.drop(["source_rank"], axis=1)

# Calculate bandwidths
bw_dir = f"{bw_results_dir}_mins"
Path(bw_dir).mkdir(parents=True, exist_ok=True)

## Inter-node bandwidths
inter_max_times = df.groupby(["data_size"], group_keys=False).apply(
    lambda x: np.max(times_inter_node(x.drop(["data_size"], axis=1).to_numpy()))
)
inter_latency = inter_max_times.loc[np.min(np.unique(df["data_size"]))].item()
inter_max_times = (inter_max_times - inter_latency)[1:]
inter_bw = inter_max_times.index / inter_max_times
inter_bw.to_csv(f"{bw_dir}/bw_inter.data", header=True)

## Intra-node bandwidths
intra_max_times = df.groupby(["data_size"], group_keys=False).apply(
    lambda x: np.max(times_intra_node(x.drop(["data_size"], axis=1).to_numpy()))
)
intra_latency = intra_max_times.loc[np.min(np.unique(df["data_size"]))].item()
intra_max_times = (intra_max_times - intra_latency)[1:]
intra_bw = intra_max_times.index / intra_max_times
intra_bw.to_csv(f"{bw_dir}/bw_intra.data", header=False)

# Produce boxplot
ax = sns.boxplot([intra_bw, inter_bw])
ax.set_xticklabels(["Intra-Node Bandwidth", "Inter-Node Bandwidth"])
ax.set_title("Intra-Node vs Inter-Node Bandwidth")
ax.set_ylabel("Bandwidth (MB/ms)")
plt.savefig("intra_inter.png")

import pandas as pd

world_size = 8

# Send Data
send_fout = open("bw_send.data", "w")
df = pd.read_csv(f"bw_results/bw_0_send.data", header=None).rename(columns={0:"data_size"}).set_index(["data_size"])
for rank in range(world_size - 1):
    df2 = pd.read_csv(f"bw_results/bw_{rank + 1}_send.data", header=None).rename(columns={0:"data_size"}).set_index(["data_size"])
    df = pd.concat([df, df2]).sort_index()
df.groupby("data_size").apply(lambda x: send_fout.write(f"{x.name}, {x.to_numpy().max()}\n"))

# Receive Data
recv_fout = open("bw_recv.data", "w")
df = pd.read_csv(f"bw_results/bw_0_recv.data", header=None).rename(columns={0:"data_size"}).set_index(["data_size"])
for rank in range(world_size - 1):
    df2 = pd.read_csv(f"bw_results/bw_{rank + 1}_recv.data", header=None).rename(columns={0:"data_size"}).set_index(["data_size"])
    df = pd.concat([df, df2]).sort_index()
df.groupby("data_size").apply(lambda x: recv_fout.write(f"{x.name}, {x.to_numpy().max()}\n"))
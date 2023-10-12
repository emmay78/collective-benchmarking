import os
import sys
import time
import torch
import argparse
import pandas as pd
import numpy as np
import itertools as it
import torch.distributed as dist
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from pathlib import Path
import itertools
from typing import Any
from enum import Enum, auto
from torch.cuda import Event
from typing import Tuple, Callable, Set, List, Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--global_rank",
        default=-1,
        type=int,
        help="global node rank for distributed training",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--job_dir",
        default="/n/home02/emyang/collective_benchmark",
        type=str,
        help="job directory",
    )
    parser.add_argument(
        "--out_dir",
        default="/n/home02/emyang/collective_benchmark/bandwidth_benchmark",
        type=str,
        help="output directory for benchmarking results",
    )
    args = parser.parse_args()

    return args


def print_env():
    print("World Size: ", os.environ["WORLD_SIZE"])
    print("Master Addr: ", os.environ["MASTER_ADDR"])
    print("Master Port:", os.environ["MASTER_PORT"])
    print("Slurm Procid: ", os.environ["SLURM_PROCID"])


class Task:
    def __init__(self) -> None:
        pass

    def __call__(self, args: Any) -> Any:

        print_env()
        print(args)

        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            self.world_size = args.world_size

        args.distributed = args.world_size > 1
        ngpus_per_node = torch.cuda.device_count()

        if "SLURM_PROCID" in os.environ:
            args.global_rank = int(os.environ["SLURM_PROCID"])
            args.local_rank = args.global_rank % ngpus_per_node

            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.global_rank,
            )

            torch.cuda.set_device(args.local_rank)

            self.experiment(args)
        return "Success"

    # Computes the rank pairs that are on the same node
    # and those that are on different nodes
    def intra_inter_idx(self):
        ngpus_per_node = torch.cuda.device_count()

        intra_pairs = []
        inter_pairs = []

        ranks = list(range(self.world_size))
        prod = list(it.product(ranks, ranks))
        for pair in prod:
            if pair[0] != pair[1]:
                if pair[0] // ngpus_per_node == pair[1] // ngpus_per_node:
                    intra_pairs.append(pair)
                else:
                    inter_pairs.append(pair)

        return (intra_pairs, inter_pairs)

    def experiment(self, args):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        send_sample_file = f"{args.out_dir}/bw_{dist.get_rank()}_send_sample.data"
        send_fout = open(send_sample_file, "w")

        dist.barrier()

        rank = dist.get_rank()

        size = 5 * (2**18)  # Initial test size is 5 MB
        size_in_mb = (size * 4) // 2**20
        discard_iters = 2

        for _ in range(discard_iters):
            self.send_recv_bench(args, size, warmup=True)

        # Perform a full all-to-all mesh communication experiment
        # and use the results to get the best inter- and intra-node
        # pair (min or max bandwidth)

        (send, recv) = self.send_recv_bench(args, size, warmup=False)
        send_list = ",".join([str(time) for time in send])
        send_fout.write(f"{send_list}\n")
        send_fout.close()

        # Rank 0 reads from every bandwidth file and computes the inter-
        # and intra-node reference pairs
        if rank == 0:
            bw_df = pd.DataFrame()
            for rank in range(self.world_size):
                print(f"{args.out_dir}/bw_{rank}_send_sample.data")
                df2 = pd.read_csv(
                    f"{args.out_dir}/bw_{rank}_send_sample.data", header=None
                )
                bw_df = pd.concat([bw_df, df2])

            [intra, inter] = self.intra_inter_idx()
            intra_times = [bw_df.iloc[idx] for idx in intra]
            inter_times = [bw_df.iloc[idx] for idx in inter]

            # Get best pairs of ranks (by MIN latency)
            intra_pair = intra[np.argmin(intra_times)]
            inter_pair = inter[np.argmin(inter_times)]

            # Write the pairs to a file so every rank can read them
            bw_bench_pairs_file = f"{args.out_dir}/bw_bench_pairs.data"
            pairs_fout = open(bw_bench_pairs_file, "w")
            pairs_fout.write(f"{intra_pair[0]},{intra_pair[1]}\n")
            pairs_fout.write(f"{inter_pair[0]},{inter_pair[1]}\n")
            pairs_fout.close()

        dist.barrier()

        bw_bench_pairs_file = f"{args.out_dir}/bw_bench_pairs.data"
        pairs_df = pd.read_csv(bw_bench_pairs_file, header=None)
        intra_pair = (pairs_df.iloc[(0, 0)], pairs_df.iloc[(0, 1)])
        inter_pair = (pairs_df.iloc[(1, 0)], pairs_df.iloc[(1, 1)])

        if rank == intra_pair[0]:
            intra_out_file = f"{args.out_dir}/bw_intra_send.data"
            intra_fout = open(intra_out_file, "w")
        if rank == inter_pair[0]:
            inter_out_file = f"{args.out_dir}/bw_inter_send.data"
            inter_fout = open(inter_out_file, "w")

        current_size = 0

        for i in range(45):
            if i == 20:
                size = 20 * (2**18)
            elif i == 30:
                size = 50 * (2**18)
            current_size += size

            size_in_mb = (current_size * 4) // 2**20

            print("size=", current_size)

            dist.barrier()

            tensor = torch.randn(current_size, device=torch.device("cuda"))
            in_tensor = torch.empty(current_size, device=torch.device("cuda"))

            if rank == intra_pair[0]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.send(tensor, dst=intra_pair[1])
                end.record()
                torch.cuda.synchronize()
                intra_fout.write(f"{size_in_mb}, {start.elapsed_time(end)}\n")
            elif rank == intra_pair[1]:
                dist.recv(in_tensor, src=intra_pair[0])
                torch.cuda.synchronize()

            dist.barrier()

        # current_size = 0
        # size = 5 * (2**18)

        # for i in range(45):
        #     if i == 20:
        #         size = 20 * (2**18)
        #     elif i == 30:
        #         size = 50 * (2**18)
        #     current_size += size

        #     size_in_mb = (current_size * 4) // 2**20

        #     print("size=", current_size)

        #     tensor = torch.randn(current_size, device=torch.device("cuda"))
        #     in_tensor = torch.empty(current_size, device=torch.device("cuda"))

        #     if rank == inter_pair[0]:
        #         start = torch.cuda.Event(enable_timing=True)
        #         end = torch.cuda.Event(enable_timing=True)
        #         start.record()
        #         dist.send(tensor, dst=inter_pair[1])
        #         end.record()
        #         torch.cuda.synchronize()
        #         inter_fout.write(f"{size_in_mb}, {start.elapsed_time(end)}\n")
        #     elif rank == inter_pair[1]:
        #         dist.recv(in_tensor, src=inter_pair[0])
        #         torch.cuda.synchronize()

        #     dist.barrier()

        if rank == intra_pair[0]:
            intra_fout.close()
        if rank == inter_pair[0]:
            inter_fout.close()

    def send_recv_bench(self, args, data_size, warmup):
        send_times = [0 for _ in range(args.world_size)]
        recv_times = [0 for _ in range(args.world_size)]

        tensor = torch.randn(data_size, device=torch.device("cuda"))
        in_tensor = torch.empty(data_size, device=torch.device("cuda"))

        dist.barrier()

        for src_rank in range(args.world_size):
            for dst_rank in range(args.world_size):
                if src_rank != dst_rank:
                    if src_rank == dist.get_rank():
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        dist.send(tensor, dst=dst_rank)
                        end.record()
                        torch.cuda.synchronize()
                        if not warmup:
                            send_times[dst_rank] = start.elapsed_time(end)
                    elif dst_rank == dist.get_rank():
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        dist.recv(in_tensor, src=src_rank)
                        end.record()
                        torch.cuda.synchronize()
                        if not warmup:
                            recv_times[src_rank] = start.elapsed_time(end)
                    dist.barrier()

        if not warmup:
            return (send_times, recv_times)


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

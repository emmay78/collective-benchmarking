import os
import sys
import time
import torch
import argparse
import torch.distributed as dist
from functools import partial
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

        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            self.world_size = args.world_size
            self.ngpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])


        args.distributed = args.world_size > 1

        if "SLURM_PROCID" in os.environ:
            args.global_rank = int(os.environ["SLURM_PROCID"])
            args.local_rank = args.global_rank % self.ngpus_per_node
            print(args)
            print("Number of GPUs per node: ", self.ngpus_per_node)
            print("Number of GPUs actually on this node:", torch.cuda.device_count())

            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.global_rank,
            )
            torch.cuda.set_device(args.local_rank)
            self.experiment(args)
            return "Success"
        return "Failed"

    def experiment(self, args):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_file = f"{args.out_dir}/bw_{args.global_rank}.data"
        fout = open(out_file, "w")
        
        current_size = 0
        size = 100 * (2 ** 10)  # Initial size is 100 KB

        # Measure latency by sending a 2-element tensor
        # latency = self.send_recv_bench(args, 1)
        # fout.write(f"0, {latency}\n")

        # Construct data size range for benchmarking
        data_sizes = []
        data_sizes.append(2 * self.world_size)
        # Exponential data sizes
        for i in range(6, 31):
            data_sizes.append(2 ** i)

        # Additional data sizes
        for i in range(44):
            if i == 1:
                size = 100 * (2 ** 10)  # increments of 100 KB
            elif i == 10:
                size = 1 * (2 ** 20)  # increments of 1 MB
            elif i == 25:
                size = 10 * (2 ** 20)  # increments of 10 MB
            elif i == 35:
                size = 100 * (2 ** 20)  # increments of 100 MB
            current_size += size
            if current_size not in data_sizes:
                data_sizes.append(current_size)


        for size in data_sizes:
            size_in_mb = size / 2**20

            times = self.send_recv_bench(args, size // (4 * self.world_size//2))
            for time in times:
                fout.write(f"{size_in_mb}, {time}\n")

        fout.close()

    def send_recv_bench(self, args, data_size):

        tensor = torch.randn(data_size, dtype= torch.float32, device=torch.cuda.current_device())
        in_tensor = torch.empty(data_size, dtype= torch.float32, device=torch.cuda.current_device())

        dist.barrier()
        src_rank = [i for i in range(self.world_size//2)]
        dst_rank = [i for i in range(self.world_size//2,self.world_size)]
        rank = dist.get_rank()
        print(f"Source rank: {src_rank} \t Dest rank: {dst_rank}")
        print(f"Data Size: {data_size} \t Current Rank: {rank}")
        # Average over three trials
        niters = 6
        times = [0 for _ in range(niters)]
        events_pre = [Event(enable_timing=True) for _ in range(niters)]
        events_post = [Event(enable_timing=True) for _ in range(niters)]


        for i in range(niters):
            if rank in src_rank :
                events_pre[i].record()
                dist.send(tensor, dst=dst_rank[rank])
                events_post[i].record()
            elif rank in dst_rank:
                events_pre[i].record()
                dist.recv(in_tensor, src=src_rank[(rank-self.world_size//2)])          
                events_post[i].record()

        torch.cuda.synchronize()
        times = [
            pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
        ]
        times = times[1:]

        return times


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

import os
import sys
import time
import torch
from torch.distributed._tensor import init_device_mesh
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
    print("Visible Cuda Devices: ", os.environ["CUDA_VISIBLE_DEVICES"])


class Task:
    def __init__(self) -> None:
        pass

    def __call__(self, args: Any) -> Any:

        print_env()

        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            self.world_size = args.world_size

        args.distributed = args.world_size > 1

        if "SLURM_PROCID" in os.environ:
            args.global_rank = int(os.environ["SLURM_PROCID"])
            self.ngpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
            args.local_rank = args.global_rank % self.ngpus_per_node
            print(args)
            print("Number of GPUs per node: ", self.ngpus_per_node)
            print("NUmber of GPUs on this node:", torch.cuda.device_count())

            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=self.world_size,
                rank=args.global_rank,
            )
            self.inter_node_pg = [None for _ in range(self.ngpus_per_node)]

            for i in range(self.ngpus_per_node):
                inter_pg = dist.new_group(ranks=[i, i+self.ngpus_per_node], backend=args.dist_backend)
                self.inter_node_pg[i] = inter_pg
            print(self.inter_node_pg)
            torch.cuda.set_device(args.local_rank)

        self.experiment(args)
        return "Success"

    def experiment(self, args):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        out_file = f"{args.out_dir}/bw_{args.global_rank}.data"
        fout = open(out_file, "w")
        
        current_size = 0
        size = 100 * (2 ** 10)  # Initial size is 100 KB

        # Construct data size range for benchmarking
        data_sizes = []
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

            time = self.all_gather_bench(args, size // (4 * self.world_size))
            fout.write(f"{size_in_mb}, {time}\n")

        fout.close()

    def all_gather_bench(self, args, data_size):
        local_pg_size = len(dist.get_process_group_ranks(self.inter_node_pg[args.local_rank]))
        print(local_pg_size)
        assert( local_pg_size == 2)
        tensor_in = torch.randn(data_size, dtype= torch.float32, device=torch.cuda.current_device()) 
        tensor_out = torch.zeros((data_size * 2), dtype= torch.float32, device=torch.cuda.current_device())

        niters = 4
        events_pre = [Event(enable_timing=True) for _ in range(niters)]
        events_post = [Event(enable_timing=True) for _ in range(niters)]

        for i in range(niters):
            events_pre[i].record()
            dist.all_gather_into_tensor(tensor_out, tensor_in, group=self.inter_node_pg[args.local_rank])
            events_post[i].record()

        torch.cuda.synchronize()
        
        times = [
                pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
            ]
        
        times = times[1:]

        return sum(times)/(niters-1)



if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

import os
import sys
import time
import torch
import argparse
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

    def experiment(self, args):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        send_data_file = f"{args.out_dir}/bw_{dist.get_rank()}_send.data"
        send_fout = open(send_data_file, "w")
        recv_data_file = f"{args.out_dir}/bw_{dist.get_rank()}_recv.data"
        recv_fout = open(recv_data_file, "w")

        dist.barrier()
        rank = dist.get_rank()

        size = 5 * (2**18)  # Initial total size is 5 MB
        niters = 10
        discard_iters = 2

        current_size = 0

        for _ in range(discard_iters):
            self.send_recv_bench(args, size, warmup=True)

        for i in range(45):
            if i == 20:
                size = 20 * (2**18)
            elif i == 30:
                size = 50 * (2**18)
            current_size += size

            size_in_mb = (current_size * 4) // 2**20

            (send, recv) = self.send_recv_bench(args, current_size, warmup=False)
            send_list = ",".join([str(time) for time in send])
            recv_list = ",".join([str(time) for time in recv])
            send_fout.write(f"{size_in_mb}, {send_list}\n")
            recv_fout.write(f"{size_in_mb}, {recv_list}\n")

            dist.barrier()

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

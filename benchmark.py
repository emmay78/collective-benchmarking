import os
import sys
import time
import torch
import argparse
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
from typing import Any
from enum import Enum, auto
from torch.cuda import Event
from typing import Tuple, Callable


class Collective(Enum):
    all_reduce = "all_reduce"
    broadcast = "broadcast"
    reduce = "reduce"
    all_gather = "all_gather"
    gather = "gather"
    scatter = "scatter"
    reduce_scatter = "reduce_scatter"
    all_to_all = "all_to_all"

    def __str__(self) -> str:
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collective",
        default=Collective.all_reduce,
        type=Collective,
        choices=list(Collective),
        help="collective function to benchmark [all_reduce, broadcast, reduce, scatter, gather, all_gather, all_to_all, reduce_scatter]",
    )
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
        default="/n/home02/emyang/collective_benchmark/benchmark_results",
        type=str,
        help="output directory for benchmarking results",
    )
    parser.add_argument(
        "--profile",
        default=False,
        type=bool,
        help="Measure with PyTorch Profiler. Disabled by default.",
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
            # self.experiment_allreduce()
        return "Success"

    def get_collective_function(self, collective_to_benchmark: Collective) -> Callable:
        if collective_to_benchmark == Collective.all_reduce:
            return dist.all_reduce
        elif collective_to_benchmark == Collective.reduce_scatter:
            return dist.reduce_scatter_tensor

    def create_tensors_all_reduce(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor = torch.randn(
            size, dtype=torch.float32, device=torch.cuda.current_device()
        )
        return (tensor,)

    def create_tensors_reduce_scatter(
        self, size: Tuple[int, ...]
    ) -> Tuple[torch.Tensor]:
        tensor_in = torch.randn(size, device=torch.cuda.current_device())
        tensor_out = torch.randn(
            size * self.world_size, device=torch.cuda.current_device()
        )
        return (tensor_in, tensor_out)

    def get_create_tensor_function(
        self, collective_to_benchmark: Collective
    ) -> Callable:
        if collective_to_benchmark == Collective.all_reduce:
            return self.create_tensors_all_reduce
        elif collective_to_benchmark == Collective.reduce_scatter:
            return self.create_tensors_reduce_scatter

    def get_collective_name(self, collective_to_benchmark: Collective) -> str:
        if collective_to_benchmark == Collective.all_reduce:
            return "all_reduce"
        elif collective_to_benchmark == Collective.reduce_scatter:
            return "reduce_scatter"

    def experiment(self, args):
        collective_function = self.get_collective_function(args.collective)
        create_args_function = self.get_create_tensor_function(args.collective)
        collective_name = self.get_collective_name(args.collective)

        warmup_iters = 10
        niters = 10
        size = 5 * (2**18)

        ######################################
        # 1. warming up CUDACachingAllocator #
        ######################################
        for _ in range(warmup_iters):
            input_args = create_args_function(size)
            collective_function(*input_args)

        # wait for all pending CUDA ops to finish
        torch.cuda.synchronize(device=args.local_rank)
        current_size = 0
        num_tasks = os.environ["WORLD_SIZE"]
        name = collective_name + f"_{num_tasks}_{args.local_rank}"
        delay_dir = f"{args.out_dir}/" + collective_name
        Path(delay_dir).mkdir(parents=True, exist_ok=True)
        fout = open(f"{delay_dir}/{name}.data", "w")
        for i in range(45):
            if i == 20:
                size = 20 * (2**18)
            elif i == 30:
                size = 50 * (2**18)
            current_size += size
            size_in_mb = (current_size * 4) // 2**20

            if args.profile:
                profile_fout = open(f"{delay_dir}/{name}.profiler.data", "w")

            ##################################################################
            # 2. measure raw delays and memory to rule out profiler overhead #
            ##################################################################
            events_pre = [Event(enable_timing=True) for _ in range(niters)]
            events_post = [Event(enable_timing=True) for _ in range(niters)]
            for i in range(niters):
                input_args = create_args_function(current_size)
                events_pre[i].record()

                if args.profile:
                    with profile(
                        activities=[ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                        use_cuda=True,
                    ) as prof:
                        collective_function(*input_args)
                else:
                    collective_function(*input_args)

                events_post[i].record()
                torch.cuda.synchronize(device=args.local_rank)

                if args.profile and args.local_rank == 0:
                    profile_fout.write(
                        prof.key_averages().table(
                            sort_by="cuda_time_total", row_limit=10
                        )
                    )
                    # Trace for viewing in Chrome profiling tool
                    # prof.export_chrome_trace(f"{delay_dir}/{name}.chrome_trace.sync")

            # wait for all pending CUDA ops to finish
            torch.cuda.synchronize(device=args.local_rank)

            delays = [
                pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
            ]

            # write results
            for delay in delays:
                print("writing results")
                fout.write(f"{size_in_mb}, {delay:.4f}\n")

            # wait for all peers to finish
            dist.barrier(device_ids=[args.local_rank])

        fout.close()
        self.teardown()
        return {
            "data_size": size_in_mb,
        }

    def teardown(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

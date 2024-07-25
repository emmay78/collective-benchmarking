import os
import sys
import time
import torch
import argparse
import torch.distributed as dist
from datetime import timedelta
from functools import partial
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from pathlib import Path
from enum import Enum, auto
from torch.cuda import Event
from typing import Any, List, Tuple, Callable
import logging


logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


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
        "--gpus_per_node",
        default=8,
        type=int,
        help="Number of GPUs per node",
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
    parser.add_argument(
        "--internode_only",
        default=False,
        type=bool,
        help="Benchmark using the internode mesh only",
    )
    parser.add_argument(
        "--async_op",
        default=False,
        type=bool,
        help="Benchmark using an asynchronous collective operation. The collective operation function returns a distributed request object on which wait() is called to block the process until completion.",
    )
    parser.add_argument(
        "--profile_size",
        default=5 * (2 ** 18),
        type=int,
        help="Data size for profile size.",
    )
    args = parser.parse_args()

    return args


def log_env():
    logger.info(f"World Size: {os.environ['WORLD_SIZE']}")
    logger.info(f"Gloabl Rank: {os.environ['RANK']}")
    logger.info(f"Local Rank: {os.environ['LOCAL_RANK']}")


class Task:
    async_collective = None

    def __init__(self) -> None:
        pass

    def __call__(self, args: Any) -> Any:
        init_logger()

        log_env()
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.global_rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

        args.distributed = args.world_size > 1
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend=args.dist_backend,
            timeout=timedelta(seconds=300)
        )
        dims: List[int] = []
        names: List[str] = []

        if args.internode_only:
            dims.append((args.world_size // args.gpus_per_node))
            names.append("dp")
            dims.append(args.gpus_per_node)
            names.append("tp")
        else:
            dims.append(args.world_size)
            names.append("dp")
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        world_mesh = init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)
        self.dp_mesh = world_mesh["dp"]
        self.dp_degree = self.dp_mesh.size()
        self.dp_rank = self.dp_mesh.get_local_rank()
        if args.internode_only:
            self.tp_mesh = world_mesh["tp"]
            self.tp_degree = self.tp_mesh.size()
            self.tp_rank = self.tp_mesh.get_local_rank()
        dist.barrier()
        if args.profile:
            self.profile(args)
        else:
            self.experiment(args)
        return "Success"

    # If the asynchronous operation for the collective is specified,
    # run the collective with async_op = True and block the current process
    # until completion
    def collective_wait(self, *input_args):
        handle = self.async_collective(*input_args, async_op=True)
        handle.wait()

    def get_collective_function(
        self, collective_to_benchmark: Collective, async_op: bool
    ) -> Callable:
        if not async_op:
            if collective_to_benchmark == Collective.all_reduce:
                return dist.all_reduce
            elif collective_to_benchmark == Collective.reduce_scatter:
                return dist.reduce_scatter_tensor
            elif collective_to_benchmark == Collective.all_to_all:
                return dist.all_to_all
            elif collective_to_benchmark == Collective.broadcast:
                return dist.broadcast
            elif collective_to_benchmark == Collective.reduce:
                return dist.reduce
            elif collective_to_benchmark == Collective.all_gather:
                return dist.all_gather_into_tensor
            elif collective_to_benchmark == Collective.gather:
                return dist.gather
        else:
            self.async_collective = self.get_collective_function(
                collective_to_benchmark, async_op=False
            )
            return self.collective_wait

    def create_tensors_all_reduce(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor = (
            torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
            + self.dp_rank * size
        )
        return (tensor,)

    def create_tensors_reduce_scatter(
        self, size: Tuple[int, ...]
    ) -> Tuple[torch.Tensor]:
        tensor_in = torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
        chunks = torch.chunk(tensor_in, self.dp_degree, dim=0)
        tensor_out = chunks[self.dp_rank].clone()

        return (tensor_out, tensor_in)

    def create_tensors_all_to_all(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor_in = (
            torch.arange(size * self.world_size, device=torch.cuda.current_device())
            + self.dp_rank * size * self.dp_degree
        )

        tensor_in = list(tensor_in.chunk(self.dp_degree))
        tensor_out = list(
            torch.empty(
                [size * self.dp_degree],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            ).chunk(self.dp_degree)
        )
        return (tensor_out, tensor_in)

    def create_tensors_broadcast(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        if self.dp_rank == 0:
            return (torch.randn(size, device=torch.cuda.current_device()), 0)
        else:
            return (torch.empty([size], device=torch.cuda.current_device()), 0)

    def create_tensors_reduce(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        return (torch.randn(size, device=torch.cuda.current_device()), 0)

    def create_tensors_all_gather(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor_out = torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
        chunks = torch.chunk(tensor_out, self.dp_degree, dim=0)
        tensor_in = chunks[self.dp_rank].clone()
        return (tensor_out, tensor_in)

    def create_tensors_gather(self, size: Tuple[int, ...]) -> Tuple[torch.Tensor]:
        tensor = (
            torch.arange(size, dtype=torch.float32, device=torch.cuda.current_device())
            + 1
            + size * self.dp_degree * self.dp_rank
        )
        gather_list = (
            [
                torch.empty(
                    [size], dtype=torch.float32, device=torch.cuda.current_device()
                )
                for _ in range(self.dp_degree)
            ]
            if self.dp_rank == 0
            else None
        )
        return (tensor, gather_list, 0)

    def get_create_tensor_function(
        self, collective_to_benchmark: Collective
    ) -> Callable:
        if collective_to_benchmark == Collective.all_reduce:
            return self.create_tensors_all_reduce
        elif collective_to_benchmark == Collective.reduce_scatter:
            return self.create_tensors_reduce_scatter
        elif collective_to_benchmark == Collective.all_to_all:
            return self.create_tensors_all_to_all
        elif collective_to_benchmark == Collective.broadcast:
            return self.create_tensors_broadcast
        elif collective_to_benchmark == Collective.reduce:
            return self.create_tensors_reduce
        elif collective_to_benchmark == Collective.all_gather:
            return self.create_tensors_all_gather
        elif collective_to_benchmark == Collective.gather:
            return self.create_tensors_gather

    def get_number_of_tensors(self, collective_to_benchmark: Collective) -> int:
        if collective_to_benchmark == Collective.all_reduce:
            return 1
        elif collective_to_benchmark == Collective.reduce_scatter:
            return 2
        elif collective_to_benchmark == Collective.all_to_all:
            return 2
        elif collective_to_benchmark == Collective.broadcast:
            return 1
        elif collective_to_benchmark == Collective.reduce:
            return 1
        elif collective_to_benchmark == Collective.all_gather:
            return 2
        elif collective_to_benchmark == Collective.gather:
            return self.dp_degree + 1

    def experiment(self, args):

        if args.internode_only:
            f_name = args.collective.__str__() + f"_{self.dp_degree * self.tp_degree}_{self.tp_rank}_{self.dp_rank}"
            data_dir = f"{args.collective.__str__()}_{self.dp_degree * self.tp_degree}"
        else:
            f_name = args.collective.__str__() + f"_{self.dp_degree}_{self.dp_rank}"
            data_dir = f"{args.collective.__str__()}_{self.dp_degree}"
        os.makedirs(os.path.join(args.out_dir, args.collective.__str__()), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, args.collective.__str__(), data_dir), exist_ok=True)
        if args.async_op:
            f_name += "_async"
        fout = open(os.path.join(args.out_dir, args.collective.__str__(), data_dir, f"{f_name}.data"), 'w')

        GiB = 2 ** 30
        MiB = 2 ** 20
        KiB = 2 ** 10

        # Get total memory available on CUDA device
        total_mem = torch.cuda.get_device_properties(0).total_memory
        total_mem -= 2 * GiB  # subtract 2 GB

        collective_function = self.get_collective_function(
            args.collective, async_op=args.async_op
        )
        create_args_function = self.get_create_tensor_function(args.collective)
        input_kwargs = {"group": self.dp_mesh.get_group()}

        warmup_iters = 3
        niters = 12
        size = 15 * KiB  # Initial size is 15 KB

        current_size = 0

        # Construct data size range for benchmarking
        data_sizes = []
        # Exponential data sizes
        for i in range(10, 29):
            data_sizes.append((2 ** i) * 15)

        # Additional data sizes
        for i in range(42):
            if i == 1:
                size = 90 * KiB  # increments of 90 KB
            elif i == 8:
                size = 15 * (MiB // 4)  # increments of ~1 MB
            elif i == 20:
                size = 15 * MiB  # increments of 15 MB
            elif i == 32:
                size = 90 * MiB  # increments of 90 MB
            current_size += size
            if current_size not in data_sizes:
                data_sizes.append(current_size)

        data_sizes.sort()
        
        for size in data_sizes:
            size_in_mb = size / MiB

            if size > (
                total_mem // self.get_number_of_tensors(args.collective)
            ):
                break
            try:
                input_args = create_args_function(size // 4)
            except torch.cuda.OutOfMemoryError:
                logger.info("Ran out of CUDA memory during warm-up")
            ######################################
            # 1. warming up CUDACachingAllocator #
            ######################################
            for _ in range(warmup_iters):
                collective_function(*input_args, **input_kwargs)

            ##################################################################
            # 2. measure raw delays and memory to rule out profiler overhead #
            ##################################################################

            events_pre = [Event(enable_timing=True) for _ in range(niters)]
            events_post = [Event(enable_timing=True) for _ in range(niters)]

            for experiment_idx in range(niters):
                    events_pre[experiment_idx].record()
                    collective_function(*input_args, **input_kwargs)
                    events_post[experiment_idx].record()

            torch.cuda.synchronize()

            delays = [
                pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
            ]
            #discard first two experiment runs
            delays = delays[2:]
            # write results
            for delay in delays:
                fout.write(f"{size_in_mb}, {delay:.4f}\n")

            # wait for all peers to finish
            dist.barrier()
        fout.flush()
        fout.close()
        dist.barrier()
        self.teardown()
        return {
            "data_size": size_in_mb,
        }

    def profile(self, args):
        collective_function = self.get_collective_function(
            args.collective, async_op=args.async_op
        )
        create_args_function = self.get_create_tensor_function(args.collective)

        num_tasks = os.environ["WORLD_SIZE"]
        name = args.collective.__str__() + f"_{num_tasks}_{dist.get_rank()}"
        data_dir = f"{args.out_dir}/{args.collective.__str__()}_{args.world_size}"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        profile_file = f"{data_dir}/{name}"
        if args.async_op:
            profile_file += "_async"
        profile_fout = open(f"{profile_file}.profiler.data", "w")

        schedule = torch.profiler.schedule(wait=1, warmup=5, active=10,)

        try:
            input_args = create_args_function(args.profile_size)
        except torch.cuda.OutOfMemoryError:
            logger.info("Ran out of CUDA memory creating tensor of size", args.profile_size)
        else:
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                schedule=schedule,
            ) as prof:
                for _ in range(15):
                    collective_function(*input_args)
                    prof.step()

        profile_fout.write(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        )

        profile_fout.close()
        self.teardown()
        size_in_mb = (args.profile_size * 4) // 2 ** 20
        return {"data_size": size_in_mb}

    def teardown(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

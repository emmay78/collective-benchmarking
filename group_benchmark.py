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
from typing import Tuple, Callable, Set, List

NUM_DISCARD = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coalesce",
        default=False,
        type=bool,
        help="Use coalescing manager. Default is False.",
    )
    parser.add_argument(
        "--num_to_coalesce",
        default=4,
        type=int,
        help="Number of tensorsto coalesce",
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
        "--data_size",
        default=5 * (2**18),
        type=int,
        help="Data size for profile size.",
    )
    args = parser.parse_args()

    return args


def print_env():
    print("World Size: ", os.environ["WORLD_SIZE"])
    print("Master Addr: ", os.environ["MASTER_ADDR"])
    print("Master Port:", os.environ["MASTER_PORT"])
    print("Slurm Procid: ", os.environ["SLURM_PROCID"])


class Task:
    async_collective = None

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
            if args.coalesce:
                self.experiment_coalesce(
                    args, [torch.Size([args.data_size]) for _ in range(args.world_size)], args.num_to_coalesce
                )
            else:
                self.experiment_base(
                    args, [torch.Size([args.data_size]) for _ in range(args.world_size)]
                )
        return "Success"

    def rank0_print(self, rank: int, *args, **kwargs):
        if rank == 0:
            print(*args, **kwargs, flush=True)

    def experiment_coalesce(
        self, args, size_per_rank: List[torch.Size], num_to_coalesce: int
    ):
        from torch.distributed.distributed_c10d import _coalescing_manager

        TO_COALESCE = True
        dist.barrier()

        rank = dist.get_rank()
        world_size = args.world_size
        self.rank0_print(rank, f"World size = {world_size}")

        dest_numel = sum(size.numel() for size in size_per_rank)
        # TODO: Assume even sizes per rank for now
        assert dest_numel % (num_to_coalesce * world_size) == 0, (
            f"dest_numel: {dest_numel}\n"
            f"num_coalesced: {num_to_coalesce}\n"
            f"world_size: {world_size}"
        )
        self.rank0_print(rank, f"[Rank 0] total numel: {dest_numel}")
        torch.manual_seed(0)
        dest_tensor_ref = torch.randn((dest_numel,), device=torch.device("cuda"))
        print(
            f"[Rank {rank}] local coalesced numel: {dest_numel // world_size}",
            flush=True,
        )
        print(
            f"[Rank {rank}] local numel: {dest_numel // world_size // num_to_coalesce}",
            flush=True,
        )
        dest_tensor = torch.empty((dest_numel,), device=torch.device("cuda"))

        times_per_all_gather = []
        outer_incr = dest_numel // num_to_coalesce
        inner_incr = outer_incr // world_size
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # offsets = list(range(0, outer_incr, inner_incr))
            start.record()
            reqs = []
            with _coalescing_manager(
                group=None, device=torch.device("cuda"), reqs=reqs
            ) if TO_COALESCE else contextlib.suppress():
                for i in range(num_to_coalesce):
                    dest_offset = outer_incr * i
                    dest_tensor_i = dest_tensor[
                        dest_offset : dest_offset + outer_incr
                    ]
                    # tensor_list = list(
                    #     torch.tensor_split(
                    #         dest_tensor[dest_offset : dest_offset + outer_incr],
                    #         offsets[1:],
                    #     )
                    # )
                    src_tensor = dest_tensor_ref[
                        dest_offset
                        + rank * inner_incr : dest_offset
                        + (rank + 1) * inner_incr
                    ]
                    if TO_COALESCE:
                        # TODO: Use `all_gather_into_tensor()` for now because
                        # the final copy in `all_gather()` is not properly
                        # blocked on in the coalescing manager.
                        # ret = dist.all_gather(
                        #     tensor_list,
                        #     src_tensor,
                        #     async_op=True,
                        # )
                        ret = dist.all_gather_into_tensor(
                            dest_tensor_i,
                            src_tensor,
                            async_op=True,
                        )
                        reqs.append(ret)
                    else:
                        dist.all_gather_into_tensor(dest_tensor_i, src_tensor)
                        # TODO: Same as above.
                        # dist.all_gather(tensor_list, src_tensor)
            for req in reqs:
                req.wait()
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            if not torch.equal(dest_tensor, dest_tensor_ref):
                torch.set_printoptions(profile="full")
                self.rank0_print(rank, torch.eq(dest_tensor, dest_tensor_ref))
                torch.set_printoptions(profile="default")
            assert torch.equal(
                dest_tensor, dest_tensor_ref
            ), f"Ref: {dest_tensor_ref}\nActual: {dest_tensor}"
            with torch.no_grad():
                dest_tensor.zero_()
            times_per_all_gather.append(elapsed_time)

        time_per_all_gather = sum(times_per_all_gather[NUM_DISCARD:]) / len(
            times_per_all_gather[NUM_DISCARD:]
        )
        self.rank0_print(
            rank,
            f"[Rank {rank}] time / coalesced all-gather ({num_to_coalesce} coalesced): {time_per_all_gather:.5f} ms",
        )
        return times_per_all_gather, time_per_all_gather

    def experiment_base(self, args, size_per_rank: List[torch.Size]):
        dist.barrier()
        rank = dist.get_rank()
        sizes: Set[torch.Size] = set()
        for size in size_per_rank:
            assert len(size) == 1, f"Expects 1D shapes but got {len(size)}D shape"
            sizes.add(size)
        assert len(sizes) == 1, f"all_gather_base() requires even input sizes"
        dest_numel = sum(size.numel() for size in size_per_rank)
        self.rank0_print(rank, f"[Rank 0] total numel: {dest_numel}")
        offsets = [0] + list(
            itertools.accumulate([size.numel() for size in size_per_rank])
        )
        torch.manual_seed(0)
        dest_tensor_ref = torch.randn((dest_numel,), device=torch.cuda.current_device())
        src_tensor = dest_tensor_ref[offsets[rank] : offsets[rank + 1]]
        print(f"[Rank {rank}] local numel: {src_tensor.numel()}", flush=True)
        dest_tensor = torch.empty((dest_numel,), device=torch.cuda.current_device())

        times_per_all_gather: List[float] = []
        for _ in range(10 + NUM_DISCARD):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_gather_into_tensor(
                dest_tensor,
                src_tensor,
            )
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            assert torch.equal(dest_tensor, dest_tensor_ref)
            with torch.no_grad():
                dest_tensor.zero_()
            times_per_all_gather.append(elapsed_time)

        time_per_all_gather = sum(times_per_all_gather[NUM_DISCARD:]) / len(
            times_per_all_gather[NUM_DISCARD:]
        )
        self.rank0_print(
            rank,
            f"[Rank {rank}] time / `_all_gather_base()`: {time_per_all_gather:.5f} ms",
        )
        return times_per_all_gather, time_per_all_gather

    # def experiment(self, args):
    #     collective_function = self.get_collective_function(
    #         args.collective, async_op=args.async_op
    #     )
    #     create_args_function = self.get_create_tensor_function(args.collective)

    #     warmup_iters = 10
    #     niters = 10
    #     size = 5 * (2**18)  # Initial size is 5 MB

    #     current_size = 0
    #     num_tasks = os.environ["WORLD_SIZE"]
    #     name = args.collective.__str__() + f"_{num_tasks}_{args.local_rank}"
    #     delay_dir = f"{args.out_dir}/" + args.collective.__str__()
    #     Path(delay_dir).mkdir(parents=True, exist_ok=True)
    #     data_file = f"{delay_dir}/{name}"
    #     if args.async_op:
    #         data_file += "_async"
    #     fout = open(f"{data_file}.data", "w")

    #     ######################################
    #     # 1. warming up CUDACachingAllocator #
    #     ######################################
    #     for _ in range(warmup_iters):
    #         input_args = create_args_function(size)
    #         collective_function(*input_args)

    #     for i in range(45):
    #         if i == 20:
    #             size = 20 * (2**18)
    #         elif i == 30:
    #             size = 50 * (2**18)
    #         current_size += size
    #         size_in_mb = (current_size * 4) // 2**20

    #         ##################################################################
    #         # 2. measure raw delays and memory to rule out profiler overhead #
    #         ##################################################################
    #         if i == 0:
    #             niters += 2

    #         events_pre = [Event(enable_timing=True) for _ in range(niters)]
    #         events_post = [Event(enable_timing=True) for _ in range(niters)]

    #         for experiment_idx in range(niters):
    #             input_args = create_args_function(current_size)
    #             events_pre[experiment_idx].record()
    #             collective_function(*input_args)
    #             events_post[experiment_idx].record()

    #         torch.cuda.synchronize()

    #         delays = [
    #             pre.elapsed_time(post) for pre, post in zip(events_pre, events_post)
    #         ]

    #         # The first experiment has a much larger CUDA time than all other experiments.
    #         # Thus, we discard the first two measurements.
    #         if i == 0:
    #             delays = delays[2:]
    #             niters -= 2

    #         # write results
    #         for delay in delays:
    #             fout.write(f"{size_in_mb}, {delay:.4f}\n")

    #         # wait for all peers to finish
    #         dist.barrier(device_ids=[args.local_rank])

    #     fout.close()
    #     self.teardown()
    #     return {
    #         "data_size": size_in_mb,
    #     }

    # def profile(self, args):
    #     collective_function = self.get_collective_function(
    #         args.collective, async_op=args.async_op
    #     )
    #     create_args_function = self.get_create_tensor_function(args.collective)

    #     num_tasks = os.environ["WORLD_SIZE"]
    #     name = args.collective.__str__() + f"_{num_tasks}_{args.local_rank}"
    #     delay_dir = f"{args.out_dir}/" + args.collective.__str__()
    #     Path(delay_dir).mkdir(parents=True, exist_ok=True)
    #     profile_file = f"{delay_dir}/{name}"
    #     if args.async_op:
    #         profile_file += "_async"
    #     profile_fout = open(f"{profile_file}.profiler.data", "w")

    #     schedule = torch.profiler.schedule(
    #         wait=1,
    #         warmup=5,
    #         active=10,
    #     )

    #     input_args = create_args_function(args.profile_size)

    #     with profile(
    #         activities=[ProfilerActivity.CUDA],
    #         record_shapes=True,
    #         profile_memory=True,
    #         schedule=schedule,
    #     ) as prof:
    #         for _ in range(15):
    #             collective_function(*input_args)
    #             prof.step()

    #     profile_fout.write(
    #         prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    #     )

    #     profile_fout.close()
    #     self.teardown()
    #     size_in_mb = (args.profile_size * 4) // 2**20
    #     return {"data_size": size_in_mb}

    # def teardown(self):
    #     dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    task = Task()
    task(args)

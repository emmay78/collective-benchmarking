#!/bin/bash

### This script submits a SLURM job for benchmark.py

#SBATCH --job-name=nccl-benchmarking
#SBATCH --partition=train
#SBATCH --time=12:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
### chdir specifies the path of the main file that you want to run using srun i.e. the path of 'benchmark.py' in our case
#SBATCH --chdir=/fsx/users/sanketpurandare/collective-benchmarking

### create a folder 'logs' in the scratch space of our lab: /n/holyscratch01/idreos_lab/Users/<your-username>/logs
### %x - specifies job-name, %j - specifies job-number, %t - specifies task number

#SBATCH --output=/fsx/users/sanketpurandare/job_logs/%x-%j.out
#SBATCH --error=/fsx/users/sanketpurandare/job_logs/%x-%j.err
#SBATCH --open-mode=append

scontrol show job $SLURM_JOBID

# create directory for logs for this SLURM job
mkdir /fsx/users/sanketpurandare/job_logs/${SLURM_JOB_ID}

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=13440
export WORLD_SIZE=24
export NUM_NODES=3

echo "JOB ID="${SLURM_JOB_ID}

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_JOB_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Username="$USER
# nccl variables
export NCCL_DEBUG='INFO'
export NCCL_DEBUG_SUBSYS='INIT,COLL,P2P,ENV,NET,TUNING,GRAPH'
export NCCL_IB_CUDA_SUPPORT=1
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
# AllReduce should always use Tree
export NCCL_ALGO='Tree'

### init virtual environment if needed
source /data/home/sanketpurandare/.bashrc
source /fsx/users/sanketpurandare/initenv.sh

# benchmarking data directory
out_dir="/fsx/users/sanketpurandare/collective-benchmarking/benchmark_results_$(date +"%Y%m%d")_$(date +"%H%M")"
bw_out_dir="/fsx/users/sanketpurandare/collective-benchmarking/bandwidth_results_$(date +"%Y%m%d")_$(date +"%H%M")"
coalescing_dir="/fsx/users/sanketpurandare/collective-benchmarking/coalescing_results_$(date +"%Y%m%d")_$(date +"%H%M")"

### Collective benchmarking
COLLECTIVES=("all_reduce")
for collective in ${COLLECTIVES[@]} 
do
    echo $collective
    srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective
    srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --async_op true
    # srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --profile true
    # srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --profile true --async_op true 
done

### Bandwidth benchmarking
echo "Bandwidh Benchmarking"
srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 bw_benchmark.py --out_dir ${bw_out_dir}
# python3 bw_calculate.py $WORLD_SIZE $NUM_NODES $bw_out_dir

# ## Coalescing manager benchmarking
# COLLECTIVES=("all_reduce" "all_gather" "reduce_scatter")
# for collective in ${COLLECTIVES[@]} 
# do
#     srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 group_benchmark.py --out_dir ${coalescing_dir} --collective $collective
#     srun --output /fsx/users/sanketpurandare/job_logs/%j/%j_%t.out --error /fsx/users/sanketpurandare/job_logs/%j/%j_%t.err python3 group_benchmark.py --out_dir ${coalescing_dir} --coalesce true --collective $collective
# done
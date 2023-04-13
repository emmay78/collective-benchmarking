#!/bin/bash

### This script submits a SLURM job for benchmark.py

#SBATCH --job-name=nccl-benchmarking
#SBATCH --partition=gpu_test
#SBATCH --time=2:00:00
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
###SBATCH --contiguous
###SBATCH --constraint="holyhdr&a100"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
###SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
### chdir specifies the path of the main file that you want to run using srun i.e. the path of 'benchmark.py' in our case

#SBATCH --chdir=/n/home02/emyang/collective_benchmark

### create a folder 'logs' in the scratch space of our lab: /n/holyscratch01/idreos_lab/Users/<your-username>/logs
### %x - specifies job-name, %j - specifies job-number, %t - specifies task number

#SBATCH --output=/n/holyscratch01/idreos_lab/Users/%u/job_logs/%x-%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/%u/job_logs/%x-%j.err
#SBATCH --open-mode=append

scontrol show job $SLURM_JOBID

# create directory for logs for this SLURM job
mkdir /n/holyscratch01/idreos_lab/Users/emyang/job_logs/${SLURM_JOB_ID}

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=8
export NUM_NODES=2

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
export NCCL_DEBUG_SUBSYS='INIT,ENV,NET'
export NCCL_IB_CUDA_SUPPORT=1

# AllReduce should always use Tree
export NCCL_ALGO='Tree'

### init virtual environment if needed
# source /n/home02/emyang/.bashrc
# source /n/idreos_lab/users/emyang/develop/initenv.sh

# benchmarking data directory
out_dir="/n/home02/emyang/collective_benchmark/benchmark_results_$(date +"%Y%m%d")_$(date +"%H%M")"
bw_out_dir="/n/home02/emyang/collective_benchmark/bandwidth_results_$(date +"%Y%m%d")_$(date +"%H%M")"
coalescing_dir="/n/home02/emyang/collective_benchmark/coalescing_results_$(date +"%Y%m%d")_$(date +"%H%M")"

### Collective benchmarking
COLLECTIVES=("all_reduce" "reduce_scatter" "all_to_all" "broadcast" "reduce" "all_gather" "gather")
for collective in ${COLLECTIVES[@]} 
do
    echo $collective
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --async_op true
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --profile true
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 benchmark.py --out_dir ${out_dir} --collective $collective --profile true --async_op true 
done

### Bandwidth benchmarking
srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 bw_benchmark.py --out_dir ${bw_out_dir}
python3 bw_calculate.py $WORLD_SIZE $NUM_NODES $bw_out_dir

## Coalescing manager benchmarking
COLLECTIVES=("all_reduce" "all_gather" "reduce_scatter")
for collective in ${COLLECTIVES[@]} 
do
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 group_benchmark.py --out_dir ${coalescing_dir} --collective $collective
    srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 group_benchmark.py --out_dir ${coalescing_dir} --coalesce true --collective $collective
done
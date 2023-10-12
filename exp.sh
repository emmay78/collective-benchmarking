#!/bin/bash

### This script submits a SLURM job for benchmark.py

#SBATCH --job-name=nccl-benchmarking
#SBATCH --partition=gpu,gpu_test
#SBATCH --time=2:00:00
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
###SBATCH --contiguous
###SBATCH --constraint="holyhdr&a100"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
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

# Master directory for all logs
LOG_DIR="/n/holyscratch01/idreos_lab/Users/emyang/job_logs"

scontrol show job $SLURM_JOBID

##### LOGGING CONFIGURATION #####
JOB_DIR="${LOG_DIR}/$(date +"%Y%m%d")_$(date +"%H%M")"
mkdir -p ${JOB_DIR}/{joblogs,collective,bandwidth,coalesce}

echo "JOB_DIR="${JOB_DIR}

# job_dir=${log_dir}/${job_creation_date_time}/${SLURM_JOB_ID}

# create directory for logs for this SLURM job
# mkdir -p ${job_dir}

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
# export WORLD_SIZE=$(scontrol show job $SLURM_JOBID | tr -s ' ' | cut -d ' ' -f 2 | awk '/TRES/ {print}' | rev | cut -d '=' -f 1 | rev)
export WORLD_SIZE=2
export NUM_NODES=$(scontrol show job $SLURM_JOBID | tr -s ' ' | cut -d ' ' -f 2 | awk '/NumNodes/ {print}' | cut -d '=' -f 2)

echo "WORLD_SIZE="${WORLD_SIZE}
echo "NUM_NODES="${NUM_NODES}

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
export NCCL_DEBUG_SUBSYS='INIT,ENV,NET,TUNING'
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_TUNING_FILE=${JOB_DIR}/init/tuning.data

# AllReduce should always use Tree
export NCCL_ALGO='Tree'

### init virtual environment if needed
# source /n/home02/emyang/.bashrc
# source /n/idreos_lab/users/emyang/develop/initenv.sh

### Collective benchmarking
# COLLECTIVES=("all_reduce" "reduce_scatter" "all_to_all" "broadcast" "reduce" "all_gather" "gather")
COLLECTIVES=("all_reduce")
for collective in ${COLLECTIVES[@]} 
do
    echo $collective
    srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective
    # srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --async_op true
    # srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --profile true
    # srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --profile true --async_op true 
done

### Bandwidth benchmarking
srun --output ${JOB_DIR}/joblogs/%j_%t_bw.out --error ${JOB_DIR}/joblogs/%j_%t_bw.err python3 bw_benchmark_two.py --out_dir ${JOB_DIR}/bandwidth
# python3 bw_calculate_fast.py ${JOB_DIR}/bandwidth

# ## Coalescing manager benchmarking
# COLLECTIVES=("all_reduce" "all_gather" "reduce_scatter")
# for collective in ${COLLECTIVES[@]} 
# do
#     srun --output ${JOB_DIR}/joblogs/%j_%t_coalesce.out --error ${JOB_DIR}/joblogs/%j_%t_coalesce.err python3 group_benchmark.py --out_dir ${JOB_DIR}/coalesce --collective $collective
#     srun --output ${JOB_DIR}/joblogs/%j_%t_coalesce.out --error ${JOB_DIR}/joblogs/%j_%t_coalesce.err python3 group_benchmark.py --out_dir ${JOB_DIR}/coalesce --coalesce true --collective $collective
# done
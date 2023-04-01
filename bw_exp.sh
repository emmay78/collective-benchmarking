#!/bin/bash

### This script submits a SLURM job for benchmark.py

#SBATCH --job-name=nccl-bw-bench
#SBATCH --partition=gpu_test
#SBATCH --time=1:00:00
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

echo "JOB ID="${SLURM_JOB_ID}

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_JOB_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "Username="$USER

#nccl variables
export NCCL_DEBUG='INFO'
export NCCL_DEBUG_SUBSYS='INIT,ENV,NET'
export NCCL_IB_CUDA_SUPPORT=1

### init virtual environment if needed
# source /n/home02/emyang/.bashrc
# source /n/idreos_lab/users/emyang/develop/initenv.sh

### the command to run
srun --output /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.out --error /n/holyscratch01/idreos_lab/Users/%u/job_logs/%j/%j_%t.err python3 bw_benchmark.py
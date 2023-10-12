#!/bin/bash

### This script submits a SLURM job for benchmark.py

#SBATCH --job-name=nccl-benchmarking
#SBATCH --partition=train
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
### chdir specifies the path of the main file that you want to run using srun i.e. the path of 'benchmark.py' in our case
#SBATCH --chdir=/fsx/users/%u/collective-benchmarking

### %x - specifies job-name, %j - specifies job-number, %t - specifies task number

#SBATCH --output=/fsx/users/%u/job_logs/%x-%j.out
#SBATCH --error=/fsx/users/%u/job_logs/%x-%j.err
#SBATCH --open-mode=append

LOG_DIR="/fsx/users/%u/job_logs"

scontrol show job $SLURM_JOBID

JOB_DIR="${LOG_DIR}/$(date +"%Y%m%d")_$(date +"%H%M")"
mkdir -p ${JOB_DIR}/{joblogs, collective, bandwidth, coalesce, init}


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=13440
export WORLD_SIZE=$(scontrol show job $SLURM_JOBID | tr -s ' ' | cut -d ' ' -f 2 | awk '/TRES/ {print}' | rev | cut -d '=' -f 1 | rev)
export NUM_NODES=$(scontrol show job $SLURM_JOBID | tr -s ' ' | cut -d ' ' -f 2 | awk '/NumNodes/ {print}' | cut -d '=' -f 2)

echo "JOB ID="${SLURM_JOB_ID}
echo "WORLD_SIZE="${WORLD_SIZE}
echo "NUM_NODES="${NUM_NODES}
### get the first node name as master address - customized for vgg slurm
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
export NCCL_TUNING_FILE=${JOB_DIR}/init/tuning.data

### init virtual environment if needed
# source /data/home/%u/.bashrc
source /fsx/users/%u/initenv.sh


### Collective benchmarking
# COLLECTIVES=("all_reduce" "reduce_scatter" "all_gather")
COLLECTIVES=("all_reduce")
for collective in ${COLLECTIVES[@]} 
do
    echo $collective
    srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective
    srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --async_op true
    # srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --profile true
    # srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 benchmark.py --out_dir ${JOB_DIR}/collective --collective $collective --profile true --async_op true 
done

### Bandwidth benchmarking
# echo "Bandwidh Benchmarking"
# srun --output ${JOB_DIR}/joblogs/%j_%t_bw.out --error ${JOB_DIR}/joblogs/%j_%t_bw.err python3 bw_benchmark_two.py --out_dir ${JOB_DIR}/bandwidth
# python3 bw_calculate.py $WORLD_SIZE $NUM_NODES $bw_out_dir

# ## Coalescing manager benchmarking
# COLLECTIVES=("all_reduce" "all_gather" "reduce_scatter")
# for collective in ${COLLECTIVES[@]} 
# do
#     srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 group_benchmark.py --out_dir ${JOB_DIR}/coalesce --collective $collective
#     srun --output ${JOB_DIR}/joblogs/%j_%t.out --error ${JOB_DIR}/joblogs/%j_%t.err python3 group_benchmark.py --out_dir ${JOB_DIR}/coalesce --coalesce true --collective $collective
# done
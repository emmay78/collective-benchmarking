export NCCL_DEBUG="INFO,WARN"
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective all_reduce
sleep 2
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective all_gather
sleep 2
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective reduce_scatter
sleep 2
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective all_reduce --gpus_per_node 2 --internode_only True
sleep 2
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective all_gather --gpus_per_node 2 --internode_only True
sleep 2
torchrun --standalone --nproc_per_node=4 benchmark.py --out_dir /data/users/sanketpurandare/collective-benchmarking/outputs --collective reduce_scatter --gpus_per_node 2 --internode_only True

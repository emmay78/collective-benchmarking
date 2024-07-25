for i in {32,24,20,16,12,8,6,5,4,3,2,1}
do
    echo "Submitting job for $i nodes"
    torchx run mast.py:train --additional_folders /data/users/sanketpurandare/collective-benchmarking --twtask_bootstrap_script run_collective_benchmarking.sh --h "grandteton" --nodes $i True
    sleep 1
done

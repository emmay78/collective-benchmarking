for i in {1..1}
do
    echo "Submitting job for $i nodes"
    sbatch --nodes=$i --dependency=singleton exp_bw_aws.sh
    sleep 1
done
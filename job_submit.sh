for i in {3..25}
do
    echo "Submitting job for $i nodes"
    sbatch --nodes=$i exp_aws.sh
    sleep 2
done
for i in {1,2,3,4,5,6,8,10,12,15,16,20}
do
    echo "Submitting job for $i nodes"
    sbatch --nodes=$i --dependency=singleton exp_aws.sh
    sleep 1
done
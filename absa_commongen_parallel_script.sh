#!/bin/sh
module load python/3.9 StdEnv/2020 gcc/9.3.0 arrow/5.0.0 && source $HOME/absa/bin/activate
for cs in 0 0.01 0.02 0.05 0.1 0.2 0.4 0.5 1
do
	echo $cs
	sbatch --time=120:00:00 --mem-per-cpu=4G --cpus-per-task=8 --account=rrg-afyshe --gres=gpu:v100l:1 --export=CS=$cs --exclude cdr2482 absa_commongen_single_script.sh
done

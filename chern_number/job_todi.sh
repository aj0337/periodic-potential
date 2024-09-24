#!/bin/bash -l

#SBATCH --job-name=test
#SBATCH --time=00:40:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=s1267
#SBATCH --uenv=prgenv-gnu/24.7:v3
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt

uenv view default

# set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

source "/users/ajayaraj/venvs/blg/bin/activate"

srun -n 24 python chern_number.py 14 24

#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --account="s1267"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anooja.jayaraj@empa.ch
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt

module load daint-mc
module load intel-oneapi/2021.3.0

srun -n 96 /users/ajayaraj/software/wannier_tools/bin/wt.x

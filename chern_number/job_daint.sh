#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --nodes=4
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

module use /apps/empa/apps/modules/all
module load daint-mc cray-python cray-fftw numpy libxc libvdwxc intel

source "/users/ajayaraj/software/gpaw/gpaw-env/bin/activate"

srun -n 48 python chern_blg.py
# srun -n 48 python chern_haldane.py

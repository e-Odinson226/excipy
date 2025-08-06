#! /bin/bash

# jobname to be used for the job allocation
#SBATCH -J ExciPy

#User notify on events (BEGIN,END,FAIL,REQUEUE,ALL)
# SBATCH --mail-type=NONE

# User to notify
# SBATCH --mail-user=

## this machine (node)  has a total 112 physical cores, with hyperthreading that makes 224 max
# task are relevant for mpi - they spawn their own processes
#SBATCH --tasks-per-node=32
# cpus per task determines how many workers openmp can get per task
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000
#SBATCH -o mybash.out
#SBATCH -e mybash.err

#conda activate.

export OMP_NUM_THREADS=1


python submit.py > outlog.out




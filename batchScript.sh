#!/bin/bash
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH -q debug
#SBATCH -t 30

. setup.sh

srun -l python ./train.py

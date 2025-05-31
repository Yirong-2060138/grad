#!/bin/bash

# EXAMPLE USAGE:
# See stat-214-gsi/computing/psc-instructions.md for guidance on how to do this on PSC.
# These are settings for a hypothetical cluster and probably won't work on PSC
# sbatch job.sh configs/default.yaml

#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 5:00:00
#SBATCH --gpus=h100-80:1

set -x

module load anaconda3
conda activate env_214
cd /ocean/projects/groupname/yasahara/code

python run_autoencoder.py configs/default.yaml
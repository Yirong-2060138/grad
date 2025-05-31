#!/bin/bash

# EXAMPLE USAGE:
# See stat-214-gsi/computing/psc-instructions.md for guidance on how to do this on PSC.
# These are settings for a hypothetical cluster and probably won't work on PSC

# sbatch job.sh configs/default.yaml

#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=low
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err 

CONFIG_PATH=$1

python run_autoencoder.py $CONFIG_PATH



#!/bin/bash

# EXAMPLE USAGE:
# See stat-214-gsi/computing/psc-instructions.md for guidance on how to do this on PSC.
# These are settings for a hypothetical cluster and probably won't work on PSC

# sbatch job_embedding.sh

#SBATCH --job-name=lab2-autoencoder
#SBATCH --partition=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err 


python get_embedding.py configs/default.yaml checkpoints/exp_0316_fine2/exp-epoch\=012-val_loss\=0.1270.ckpt 
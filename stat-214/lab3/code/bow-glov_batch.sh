#!/bin/bash
#SBATCH --account=mth240012p
#SBATCH --job-name=bow_embed
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --partition=GPU-shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=5
#SBATCH -o bow_embed.out
#SBATCH -e bow_embed.err

module load anaconda3
conda activate env_214

python embedding_GloVe.py



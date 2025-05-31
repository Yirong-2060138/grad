#!/bin/bash
#SBATCH --account=mth240012p
#SBATCH --job-name=run_ridge
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=5
#SBATCH --time=07:00:00
#SBATCH -o ridge.out
#SBATCH -e ridge.err

module load anaconda3
conda activate env_214

export PYTHONPATH=$PYTHONPATH:/ocean/projects/mth240012p/sapountz/lab3_fmri/code

python rdge-scrpt.py

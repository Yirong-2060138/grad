#!/bin/bash

# ========================================
# Run script for STAT-214 Lab 2
# ========================================

conda activate stat214

## Autoencoder

echo "=== Running Autoencoder Training ==="
python run_autoencoder.py configs/default.yaml

echo "=== Autoencoder Finetuning==="
python fine_tuning.py configs/finetuning.yaml checkpoints/exp_0316/exp-epoch=036-val_loss=0.1052.ckpt

echo "=== Generating Embeddings ==="
python get_embedding.py \
  configs/default.yaml \
  checkpoints/exp_0316_fine2/exp-epoch=012-val_loss=0.1270.ckpt

## Modeling

echo "=== Running Classical and Deep Models ==="
python models/helper.py
python models/logistic_regression.py
python models/svm.py
python models/qda.py
python models/neural_network.py
python models/XGboost_model.py


echo "=== All steps completed. ==="

# Deactivate environment
conda deactivate



#!/bin/bash

# Run embedding extraction
echo "Running extract_bert_embeddings.py..."
python extract_bert_embeddings.py

# Run ridge regression
echo "Running run_ridge_regression.py..."
python run_ridge_regression.py

# Run BERT fine-tuning
echo "Running finetune.py..."
python finetune.py

echo "All tasks completed."

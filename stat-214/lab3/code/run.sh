#!/bin/bash

set -e 

# Activate your conda environment
conda activate lab3-fmri

# Run embedding scripts
echo "Running BOW embedding..."
python code/embedding_bow.py

echo "Running GloVe embedding..."
python code/embedding_GloVe.py

echo "Running Word2Vec embedding..."
python code/embedding_word2vec.py

echo "All embedding scripts completed successfully!"
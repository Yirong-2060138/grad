#!/bin/bash

set -e

conda activate stat214

echo "Running data cleaning..."
python clean.py

echo "Generating figures..."
jupyter nbconvert --to notebook --execute --inplace eda.ipynb

echo "Running modeling..."
jupyter nbconvert --to notebook --execute --inplace model.ipynb

echo "All scripts executed successfully!"

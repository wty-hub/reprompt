#!/bin/bash

# Activate the conda environment
# We need to source conda.sh first to use conda activate in scripts
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh

# Activate the prompt environment
conda activate prompt

# Run the Python script with the specified dataset
python pit.py --dataset mmimdb

# Optional: Print a completion message
echo "Execution complete."
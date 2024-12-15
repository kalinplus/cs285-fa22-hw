#!/bin/bash

# Define the arrays for batch sizes and learning rates
batch_sizes=(1000 3000 5000)
learning_rates=(0.01 0.05 0.1)

# Loop over all combinations of batch sizes and learning rates
for b in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    # Construct the experiment name
    exp_name="q2_b${b}_r${lr}"

    # Run the Python script with the current parameters
    python cs285/scripts/run_hw2.py \
      --env_name InvertedPendulum-v4 \
      --ep_len 1000 \
      --discount 0.9 \
      -n 100 \
      -l 2 \
      -s 64 \
      -b "$b" \
      -lr "$lr" \
      -rtg \
      --exp_name "$exp_name"
  done
done
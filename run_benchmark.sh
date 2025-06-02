#!/bin/bash

# Usage:
#   ./run_batch.sh 0 16          # Run from index 0 to 15
#   ./run_batch.sh 10 20 --metric # Optional: add extra args to pass to script

START=${1:-0}
END=${2:-9999}  # default: very large to include all
EXTRA_ARGS=${@:3}

CONFIG_DIR="configs/wild-ti2i"
SCRIPT="run_demo_fpe_norm_benchmark.py"  

# Collect all config paths and slice them
CONFIGS=($(ls ${CONFIG_DIR}/*.json | sort))
SELECTED_CONFIGS=("${CONFIGS[@]:$START:$((END - START))}")

echo "Running ${#SELECTED_CONFIGS[@]} config(s) from index $START to $((END - 1))"

for CONFIG in "${SELECTED_CONFIGS[@]}"; do
    echo "Running config: $CONFIG"
    python $SCRIPT --config_file "$CONFIG" $EXTRA_ARGS
done

#!/bin/bash

# Usage:
#   ./grid_search.sh 0 16 --metric

START=${1:-0}
END=${2:-9999}
EXTRA_ARGS=${@:3}

PARENT_CONFIG_DIR="configs"
BENCHMARK_SCRIPT="run_demo_fpe_norm_benchmark.py"
RESULT_SCRIPT="accumulate_benchmark_result.py"

# Get sorted config directories
CONFIG_DIRS=($(find "$PARENT_CONFIG_DIR" -maxdepth 1 -mindepth 1 -type d -name "wild-ti2i_*"| sort))
SELECTED_DIRS=("${CONFIG_DIRS[@]:$START:$((END - START))}")

echo "Running benchmark on ${#SELECTED_DIRS[@]} config dir(s) from index $START to $((END - 1))"

for CONFIG_DIR in "${SELECTED_DIRS[@]}"; do
    DIR_NAME=$(basename "$CONFIG_DIR")
    echo "📂 Processing config dir: $DIR_NAME"

    CONFIGS=($(ls "$CONFIG_DIR"/*.json 2>/dev/null | sort))

    if [ ${#CONFIGS[@]} -eq 0 ]; then
        echo "  ⚠️  No JSON config files found in $CONFIG_DIR"
        continue
    fi

    for CONFIG in "${CONFIGS[@]}"; do
        echo "  🚀 Running config: $CONFIG"
        python "$BENCHMARK_SCRIPT" --config_file "$CONFIG" $EXTRA_ARGS
    done

    # Run result accumulation after all configs in this dir are done
    for VERSION in tome standard; do
        echo "  📊 Accumulating results for version: $VERSION"
        python "$RESULT_SCRIPT" --dataset_name "$DIR_NAME" --version "$VERSION"
    done
done

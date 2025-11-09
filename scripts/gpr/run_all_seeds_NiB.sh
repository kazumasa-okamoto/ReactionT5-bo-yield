#!/bin/bash
# Run GPR experiments with multiple seeds for NiB dataset

# Configuration
DATA_PATH="../../data/NiB/inchi_23l_reaction_t5_ready.csv"
DATASET_NAME="NiB"
N_TRIALS=100
RADIUS=4
N_BITS=2048
OUTPUT_DIR="../../runs"

# Seeds to run
SEEDS=(1 2 3 4 5)

echo "=========================================="
echo "Running GPR experiments with multiple seeds"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Data path: $DATA_PATH"
echo "N trials: $N_TRIALS"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Run experiment for each seed
for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting experiment with seed: $SEED"
    echo "=========================================="

    python run_experiment.py \
        --data "$DATA_PATH" \
        --dataset-name "$DATASET_NAME" \
        --n-trials $N_TRIALS \
        --radius $RADIUS \
        --n-bits $N_BITS \
        --output-dir "$OUTPUT_DIR" \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "[Success] Experiment with seed $SEED completed successfully"
    else
        echo "[Failure] Experiment with seed $SEED failed"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"

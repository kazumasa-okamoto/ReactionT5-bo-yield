#!/bin/bash
# Run Optuna TPE experiments with multiple seeds for Suzuki-Miyaura dataset

# Configuration
DATA_PATH="../../data/Suzuki-Miyaura/aap9112_complete_grid.csv"
DATASET_NAME="SM"
N_TRIALS=100
N_STARTUP_TRIALS=10
N_EI_CANDIDATES=24
OUTPUT_DIR="../../runs"

# Seeds to run
SEEDS=(1 2 3 4 5)

echo "=========================================="
echo "Running Optuna TPE experiments with multiple seeds"
echo "=========================================="
echo "Dataset: $DATASET_NAME (Suzuki-Miyaura)"
echo "Data path: $DATA_PATH"
echo "N trials: $N_TRIALS"
echo "N startup trials: $N_STARTUP_TRIALS"
echo "N EI candidates: $N_EI_CANDIDATES"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Run experiment for each seed
for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting experiment with seed: $SEED"
    echo "=========================================="

    python run_experiment_SM.py \
        --data "$DATA_PATH" \
        --n-trials $N_TRIALS \
        --seed $SEED \
        --output-dir "$OUTPUT_DIR" \
        --n-startup-trials $N_STARTUP_TRIALS \
        --n-ei-candidates $N_EI_CANDIDATES

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

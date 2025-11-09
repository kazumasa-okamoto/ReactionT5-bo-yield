#!/bin/bash
# Run experiments with multiple seeds

# Configuration
DATA_PATH="../../data/NiB/inchi_23l_reaction_t5_ready.csv"
DATASET_NAME="NiB"
N_ROUNDS=10
TRIALS_PER_ROUND=10
N_MC_SAMPLES=10
BATCH_SIZE_PREDICTION=64
LEARNING_RATE=5e-4
EPOCHS_PER_ROUND=2
WEIGHT_DECAY=0.01
BATCH_SIZE_TRAIN=8
BATCH_SIZE_EVAL=16
VAL_RATIO=0.2
OUTPUT_DIR="../../runs"

# Seeds to run
SEEDS=(1 2 3 4 5)

echo "=========================================="
echo "Running experiments with multiple seeds"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Data path: $DATA_PATH"
echo "N rounds: $N_ROUNDS"
echo "Trials per round: $TRIALS_PER_ROUND"
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
        --n-rounds $N_ROUNDS \
        --trials-per-round $TRIALS_PER_ROUND \
        --n-mc-samples $N_MC_SAMPLES \
        --batch-size-prediction $BATCH_SIZE_PREDICTION \
        --learning-rate $LEARNING_RATE \
        --epochs-per-round $EPOCHS_PER_ROUND \
        --weight-decay $WEIGHT_DECAY \
        --batch-size-train $BATCH_SIZE_TRAIN \
        --batch-size-eval $BATCH_SIZE_EVAL \
        --val-ratio $VAL_RATIO \
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

# Run visualization for all seeds
echo ""
echo "=========================================="
echo "Generating visualizations..."
echo "=========================================="

for SEED in "${SEEDS[@]}"
do
    EXP_NAME="bayes_${N_ROUNDS}rounds_${TRIALS_PER_ROUND}trials_${DATASET_NAME}_seed${SEED}"
    CSV_PATH="$OUTPUT_DIR/$EXP_NAME/optimization_log.csv"
    LOGDIR="$OUTPUT_DIR/$EXP_NAME"

    if [ -f "$CSV_PATH" ]; then
        echo "Visualizing seed $SEED..."
        python visualize_results.py --csv "$CSV_PATH" --logdir "$LOGDIR"
        echo "[Success] Visualization for seed $SEED completed"
    else
        echo "[Warning] CSV file not found for seed $SEED: $CSV_PATH"
    fi
done

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "=========================================="

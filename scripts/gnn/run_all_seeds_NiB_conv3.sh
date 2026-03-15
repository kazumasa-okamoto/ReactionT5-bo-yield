#!/bin/bash
# Run GNN experiments with multiple seeds for NiB dataset (3 conv layers)

# Configuration
DATA_PATH="data/NiB/inchi_23l_reaction_t5_ready.csv"
DATASET_NAME="NiB"
N_INITIAL_RANDOM=10
N_BO_ITERATIONS=90
N_MC_SAMPLES=10
LEARNING_RATE=1e-3
NUM_EPOCHS=100
WEIGHT_DECAY=1e-4
BATCH_SIZE=32
VAL_RATIO=0.2
HIDDEN_DIM=32
DROPOUT_RATE=0.2
NUM_CONV_LAYERS=3
OUTPUT_DIR="runs"

# Seeds to run
SEEDS=(1 2 3 4 5)

echo "=========================================="
echo "Running GNN experiments with multiple seeds"
echo "=========================================="
echo "Dataset: $DATASET_NAME (Nickel-catalyzed Borylation)"
echo "Conv layers: $NUM_CONV_LAYERS"
echo "Data path: $DATA_PATH"
echo "Initial random samples: $N_INITIAL_RANDOM"
echo "BO iterations: $N_BO_ITERATIONS"
echo "Total trials: $((N_INITIAL_RANDOM + N_BO_ITERATIONS))"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Run experiment for each seed
for SEED in "${SEEDS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Starting experiment with seed: $SEED"
    echo "=========================================="

    python scripts/gnn/run_experiment.py \
        --data "$DATA_PATH" \
        --dataset-name "$DATASET_NAME" \
        --n-initial-random $N_INITIAL_RANDOM \
        --n-bo-iterations $N_BO_ITERATIONS \
        --n-mc-samples $N_MC_SAMPLES \
        --learning-rate $LEARNING_RATE \
        --num-epochs $NUM_EPOCHS \
        --weight-decay $WEIGHT_DECAY \
        --batch-size $BATCH_SIZE \
        --val-ratio $VAL_RATIO \
        --hidden-dim $HIDDEN_DIM \
        --dropout-rate $DROPOUT_RATE \
        --num-conv-layers $NUM_CONV_LAYERS \
        --output-dir "$OUTPUT_DIR" \
        --seed $SEED

    if [ $? -eq 0 ]; then
        echo "✓ Experiment with seed $SEED completed successfully"
    else
        echo "✗ Experiment with seed $SEED failed"
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
    TOTAL_TRIALS=$((N_INITIAL_RANDOM + N_BO_ITERATIONS))
    EXP_NAME="gnn_bo_${TOTAL_TRIALS}trials_conv${NUM_CONV_LAYERS}_${DATASET_NAME}_seed${SEED}"
    CSV_PATH="$OUTPUT_DIR/$EXP_NAME/optimization_log.csv"
    LOGDIR="$OUTPUT_DIR/$EXP_NAME"

    if [ -f "$CSV_PATH" ]; then
        echo "Visualizing seed $SEED..."
        python scripts/gnn/visualize_results.py --csv "$CSV_PATH" --logdir "$LOGDIR"
        echo "✓ Visualization for seed $SEED completed"
    else
        echo "⚠ CSV file not found for seed $SEED: $CSV_PATH"
    fi
done

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "=========================================="

# Bayesian Optimization Experiment Scripts

This directory (`scripts/bo_yield/`) contains scripts for running Bayesian optimization experiments with fine-tuning for reaction yield prediction.

## Location

All scripts are located in `scripts/bo_yield/` directory. Run commands from this directory:
```bash
cd scripts/bo_yield
```

## Files

- `data_utils.py`: Data processing utilities
  - **Dataset-aware preprocessing**: NiB vs other datasets (SM, BH)
  - SMILES canonicalization with RDKit
  - Reaction dictionary creation
- `model_utils.py`: Model and MC Dropout utilities
  - ReactionT5Yield model
  - MC Dropout for uncertainty estimation
- `training_utils.py`: Training utilities (Dataset, Trainer, Collator)
- `bayesian_optimizer.py`: Bayesian optimization with fine-tuning implementation
  - Expected Improvement acquisition function
  - Cumulative fine-tuning
- `run_experiment.py`: Main experiment script
- `visualize_results.py`: Visualization utilities
  - CSV log visualization (parity plots, error analysis, etc.)
  - TensorBoard log visualization
- `run_all_seeds.sh`: Shell script to run experiments with multiple seeds
- `run_all_seeds.py`: Python script to run experiments with multiple seeds

## Data Preprocessing

The preprocessing differs based on dataset type:

### NiB Dataset
- **REACTANT**: Canonicalized and sorted
- **PRODUCT**: Canonicalized and sorted
- **REAGENT**: Combined from `CATALYST + REAGENT + SOLVENT` (NOT canonicalized)
  - Example: `catalyst.reagent.solvent`

### Other Datasets (SM, BH, etc.)
- **REACTANT**: Canonicalized and sorted
- **REAGENT**: Canonicalized and sorted (treated as-is from CSV)
- **PRODUCT**: Canonicalized and sorted

**Why different preprocessing?**
- NiB: The dataset structure requires combining multiple columns into REAGENT
- SM/BH: REAGENT is already provided as a single SMILES string

## Usage

### Single Experiment

Run a single experiment with a specific seed:

```bash
cd scripts/bo_yield
python run_experiment.py \
    --data ../../data/NiB/inchi_23l_reaction_t5_ready.csv \
    --dataset-name NiB \
    --n-rounds 10 \
    --trials-per-round 10 \
    --seed 42
```

### Multiple Seeds

Run experiments with multiple seeds (1, 2, 3, 4, 5):

**Using bash script:**
```bash
cd scripts/bo_yield
bash run_all_seeds.sh
```

**Using Python script:**
```bash
cd scripts/bo_yield
python run_all_seeds.py  # Defaults to seeds 1 2 3 4 5
# Or pass explicit values
python run_all_seeds.py --seeds 1 2 3 4 5
```

### Visualization Only

Generate visualizations for existing experiment results:

```bash
python visualize_results.py \
    --csv ../../runs/bayes_10rounds_10trials_NiB_seed1/optimization_log.csv \
    --logdir ../../runs/bayes_10rounds_10trials_NiB_seed1
```

## Command Line Arguments

### run_experiment.py

**Data arguments:**
- `--data`: Path to CSV data file (required)
- `--dataset-name`: Dataset name for output directory (default: "NiB")

**Optimization arguments:**
- `--n-rounds`: Number of optimization rounds (default: 10)
- `--trials-per-round`: Number of trials per round (default: 10)
- `--n-mc-samples`: Number of MC Dropout samples (default: 10)
- `--batch-size-prediction`: Batch size for prediction (default: 64)

**Fine-tuning arguments:**
- `--learning-rate`: Learning rate (default: 5e-4)
- `--epochs-per-round`: Training epochs per round (default: 2)
- `--weight-decay`: Weight decay (default: 0.01)
- `--batch-size-train`: Batch size for training (default: 8)
- `--batch-size-eval`: Batch size for evaluation (default: 16)
- `--val-ratio`: Validation split ratio (default: 0.2)

**Model arguments:**
- `--model-name`: Model name or path (default: "sagawa/ReactionT5v2-yield")
- `--max-length`: Max sequence length (default: 512)

**Output arguments:**
- `--output-dir`: Base output directory (default: "runs")
- `--no-checkpoints`: Do not save model checkpoints

**Reproducibility:**
- `--seed`: Random seed (default: 42)

### run_all_seeds.py

Same arguments as `run_experiment.py`, plus:
- `--seeds`: List of random seeds to run (default: [1, 2, 3, 4, 5])
- `--skip-visualization`: Skip visualization step

### visualize_results.py

- `--csv`: Path to optimization log CSV (required)
- `--logdir`: TensorBoard log directory (optional)
- `--output`: Output directory (optional)
- `--show`: Show plots interactively
- `--dpi`: Image resolution (default: 180)

## Output Structure

Each experiment creates a directory with the following structure:

```
ReactionT5-bo-yield/
|-- scripts/
|   `-- bo_yield/                   # Bayesian optimization scripts (this directory)
|       |-- run_experiment.py
|       |-- run_all_seeds.py
|       |-- run_all_seeds.sh
|       `-- ...
|-- data/
|   |-- NiB/
|   |   `-- inchi_23l_reaction_t5_ready.csv
|   |-- SM/
|   `-- BH/
`-- runs/                           # Experiment results
    |-- bayes_10rounds_10trials_NiB_seed1/
    |   |-- config.json                 # Experiment configuration
    |   |-- summary.json                # Summary statistics
    |   |-- optimization_log.csv        # Detailed optimization log
    |   |-- round_1/                    # Fine-tuning checkpoint (round 1)
    |   |   |-- config.json
    |   |   |-- pytorch_model.bin
    |   |   `-- logs/                   # TensorBoard logs
    |   |-- round_2/                    # Fine-tuning checkpoint (round 2)
    |   |   `-- ...
    |   |-- ...
    |   |-- visualization/              # Visualization plots
    |   |   |-- parity.png
    |   |   |-- error_hist.png
    |   |   |-- best_true_per_round.png
    |   |   |-- optimization_progress.png
    |   |   `-- ...
    |   `-- tensorboard_visualization/  # TensorBoard plots
    |       |-- training_loss_all_rounds.png
    |       |-- validation_loss_all_rounds.png
    |       `-- ...
    |-- bayes_10rounds_10trials_NiB_seed2/
    |-- bayes_10rounds_10trials_NiB_seed3/
    `-- ...
```

## Customization

### Modifying run_all_seeds.sh

Edit the configuration variables at the top of the script:

```bash
DATA_PATH="../../data/NiB/inchi_23l_reaction_t5_ready.csv"
DATASET_NAME="NiB"
N_ROUNDS=10
TRIALS_PER_ROUND=10
SEEDS=(1 2 3 4 5)
```

### Running on Different Datasets

To run on a different dataset (e.g., Suzuki-Miyaura or Buchwald-Hartwig):

**IMPORTANT:** The `--dataset-name` parameter affects data preprocessing:
- `NiB`: REAGENT is combined from CATALYST + REAGENT + SOLVENT (no canonicalization of REAGENT)
- Other datasets (e.g., `SM`, `BH`): REACTANT, REAGENT, and PRODUCT are all canonicalized

```bash
# Suzuki-Miyaura dataset
python run_experiment.py \
    --data ../../data/SM/suzuki_miyaura_data.csv \
    --dataset-name SM \
    --seed 1

# Buchwald-Hartwig dataset
python run_experiment.py \
    --data ../../data/BH/buchwald_hartwig_data.csv \
    --dataset-name BH \
    --seed 1
```

For multiple seeds:
```bash
# Using run_all_seeds.py for SM dataset
python run_all_seeds.py \
    --data ../../data/SM/suzuki_miyaura_data.csv \
    --dataset-name SM \
    --seeds 1 2 3 4 5
```

## Notes

- GPU is recommended for faster training and prediction
- Each round of fine-tuning saves a checkpoint (disable with `--no-checkpoints`)
- Visualizations are automatically generated after all experiments complete
- TensorBoard logs can be viewed with: `tensorboard --logdir runs/`

## Dependencies

All dependencies are managed via pyproject.toml. Key dependencies:
- torch
- transformers
- rdkit
- pandas
- numpy
- scipy
- matplotlib
- tensorboard

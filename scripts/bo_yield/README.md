# Bayesian Optimization with ReactionT5

Scripts for Bayesian optimization using ReactionT5 + MC Dropout with cumulative fine-tuning for reaction yield prediction.

> All commands are run from the **project root**.

## Files

- `data_utils.py`: Data loading and SMILES canonicalization (dataset-aware preprocessing)
- `model_utils.py`: ReactionT5Yield model and MC Dropout inference
- `training_utils.py`: Dataset, Trainer, and Collator for fine-tuning
- `bayesian_optimizer.py`: Bayesian optimization loop (Expected Improvement, cumulative fine-tuning)
- `run_experiment.py`: Single experiment script
- `run_all_seeds.py`: Run experiments over multiple seeds
- `visualize_results.py`: Generate plots from optimization log CSV and TensorBoard logs
- `run_all_seeds_BH.sh`: Run BH experiments with seeds 1–5
- `run_all_seeds_NiB.sh`: Run NiB experiments with seeds 1–5
- `run_all_seeds_SM.sh`: Run SM experiments with seeds 1–5

## Usage

### Single Experiment

```bash
python scripts/bo_yield/run_experiment.py \
    --data data/NiB/inchi_23l_reaction_t5_ready.csv \
    --dataset-name NiB \
    --n-rounds 10 \
    --trials-per-round 10 \
    --seed 42
```

### Multiple Seeds

```bash
# Shell scripts (seeds 1–5)
bash scripts/bo_yield/run_all_seeds_BH.sh
bash scripts/bo_yield/run_all_seeds_NiB.sh
bash scripts/bo_yield/run_all_seeds_SM.sh

# Python script
python scripts/bo_yield/run_all_seeds.py \
    --data data/NiB/inchi_23l_reaction_t5_ready.csv \
    --dataset-name NiB \
    --seeds 1 2 3 4 5
```

### Visualization Only

```bash
python scripts/bo_yield/visualize_results.py \
    --csv runs/bayes_10rounds_10trials_NiB_seed1/optimization_log.csv \
    --logdir runs/bayes_10rounds_10trials_NiB_seed1
```

## Command Line Arguments

### run_experiment.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to CSV data file |
| `--dataset-name` | `"NiB"` | Dataset name (affects preprocessing) |
| `--n-rounds` | `10` | Number of optimization rounds |
| `--trials-per-round` | `10` | Trials per round |
| `--n-mc-samples` | `10` | MC Dropout samples |
| `--batch-size-prediction` | `64` | Batch size for prediction |
| `--learning-rate` | `5e-4` | Fine-tuning learning rate |
| `--epochs-per-round` | `2` | Training epochs per round |
| `--weight-decay` | `0.01` | Weight decay |
| `--batch-size-train` | `8` | Batch size for training |
| `--batch-size-eval` | `16` | Batch size for evaluation |
| `--val-ratio` | `0.2` | Validation split ratio |
| `--model-name` | `"sagawa/ReactionT5v2-yield"` | Model name or path |
| `--max-length` | `512` | Max sequence length |
| `--output-dir` | `"runs"` | Base output directory |
| `--no-checkpoints` | `False` | Disable checkpoint saving |
| `--seed` | `42` | Random seed |

### run_all_seeds.py

Same as `run_experiment.py`, plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | `[1,2,3,4,5]` | List of seeds to run |
| `--skip-visualization` | `False` | Skip visualization step |

### visualize_results.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to optimization log CSV |
| `--logdir` | `None` | TensorBoard log directory |
| `--output` | `None` | Output directory |
| `--show` | `False` | Show plots interactively |
| `--dpi` | `180` | Image resolution |

## Data Preprocessing

Preprocessing differs by dataset:

**NiB**: REACTANT and PRODUCT are canonicalized. REAGENT is combined from `CATALYST + REAGENT + SOLVENT` without pre-canonicalization, then canonicalized component-wise.

**SM / BH**: REACTANT, REAGENT, and PRODUCT are all canonicalized first, then REAGENT is combined from `CATALYST + REAGENT + SOLVENT` and canonicalized component-wise.

## Output

Output directory name: `bayes_{n_rounds}rounds_{trials_per_round}trials_{dataset}_seed{seed}`

```
runs/
└── bayes_10rounds_10trials_NiB_seed1/
    ├── config.json
    ├── summary.json
    ├── optimization_log.csv
    ├── round_1/                    # Fine-tuning checkpoint
    │   ├── pytorch_model.bin
    │   └── logs/                   # TensorBoard logs
    ├── round_2/
    └── visualization/
        ├── parity.png
        ├── error_hist.png
        ├── best_true_per_round.png
        ├── uncertainty_vs_yield.png
        ├── calibration.png
        ├── round_box.png
        ├── error_box_by_round.png
        ├── error_by_round.png
        ├── round_best.png
        ├── acquisition_vs_yield.png
        └── optimization_progress.png
```

**optimization_log.csv columns:**
`timestamp`, `round`, `trial`, `reactant`, `reagent`, `product`, `reaction_smiles`,
`predicted_mean`, `predicted_std`, `actual_yield`, `error_pct`, `acquisition_value`,
`was_used_for_ft`, `round_best_pred`, `round_best_true`

## Notes

- GPU is recommended for faster training and prediction
- Checkpoints are saved each round (disable with `--no-checkpoints`)
- TensorBoard logs: `tensorboard --logdir runs/`

## Dependencies

Managed via `pyproject.toml`. Key packages: `torch`, `transformers`, `rdkit`, `scipy`, `matplotlib`, `tensorboard`

# GNN-based Bayesian Optimization

Scripts for Bayesian optimization using a Graph Neural Network (GNN) + MC Dropout for reaction yield prediction. The GNN is retrained from scratch at each iteration using cumulative observed data.

> All commands are run from the **project root**.

## Files

- `data_utils.py`: Data loading, SMILES canonicalization, and PyG graph construction
- `model_utils.py`: GNN model definition and MC Dropout inference
- `bayesian_optimizer.py`: Bayesian optimization loop (Expected Improvement, from-scratch retraining)
- `run_experiment.py`: Single experiment script
- `run_all_seeds.py`: Run experiments over multiple seeds
- `visualize_results.py`: Generate plots from optimization log CSV
- `run_all_seeds_BH.sh`: Run BH experiments with seeds 1–5 (1 conv layer)
- `run_all_seeds_NiB.sh`: Run NiB experiments with seeds 1–5 (1 conv layer)
- `run_all_seeds_SM.sh`: Run SM experiments with seeds 1–5 (1 conv layer)
- `run_all_seeds_BH_conv2.sh`: Run BH experiments with 2 conv layers
- `run_all_seeds_BH_conv3.sh`: Run BH experiments with 3 conv layers
- `run_all_seeds_NiB_conv2.sh`, `run_all_seeds_NiB_conv3.sh`: NiB variants
- `run_all_seeds_SM_conv2.sh`, `run_all_seeds_SM_conv3.sh`: SM variants

## Usage

### Single Experiment

```bash
python scripts/gnn/run_experiment.py \
    --data data/Buchwald-Hartwig/Dreher_and_Doyle_reaction_t5_ready.csv \
    --dataset-name BH \
    --n-initial-random 10 \
    --n-bo-iterations 90 \
    --seed 42
```

### Multiple Seeds

```bash
# Shell scripts (seeds 1–5)
bash scripts/gnn/run_all_seeds_BH.sh
bash scripts/gnn/run_all_seeds_NiB.sh
bash scripts/gnn/run_all_seeds_SM.sh

# Python script
python scripts/gnn/run_all_seeds.py \
    --data data/Buchwald-Hartwig/Dreher_and_Doyle_reaction_t5_ready.csv \
    --dataset-name BH \
    --seeds 1 2 3 4 5
```

### Visualization Only

```bash
python scripts/gnn/visualize_results.py \
    --csv runs/gnn_bo_100trials_conv1_BH_seed1/optimization_log.csv \
    --logdir runs/gnn_bo_100trials_conv1_BH_seed1
```

## Command Line Arguments

### run_experiment.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to CSV data file |
| `--dataset-name` | `"BH"` | Dataset name (affects preprocessing) |
| `--graph-data-dir` | `None` | Directory to save/load graph data (auto-generated if not set) |
| `--n-initial-random` | `10` | Initial random samples |
| `--n-bo-iterations` | `90` | Bayesian optimization iterations |
| `--n-mc-samples` | `10` | MC Dropout samples |
| `--learning-rate` | `1e-3` | Learning rate |
| `--num-epochs` | `100` | Training epochs per iteration |
| `--weight-decay` | `1e-4` | Weight decay |
| `--batch-size` | `32` | Batch size |
| `--val-ratio` | `0.2` | Validation split ratio |
| `--early-stopping-patience` | `10` | Early stopping patience (0 to disable) |
| `--hidden-dim` | `256` | GNN hidden dimension |
| `--dropout-rate` | `0.2` | Dropout rate |
| `--num-conv-layers` | `1` | Number of graph conv layers (1, 2, or 3) |
| `--output-dir` | `"runs"` | Base output directory |
| `--save-checkpoints` | `False` | Save model checkpoints |
| `--seed` | `42` | Random seed |

### run_all_seeds.py

Same as `run_experiment.py`, plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--seeds` | `[1,2,3,4,5]` | List of seeds to run |

### visualize_results.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to optimization log CSV |
| `--logdir` | `None` | Output directory |
| `--show` | `False` | Show plots interactively |
| `--dpi` | `180` | Image resolution |

## Data Preprocessing

Same dataset-aware preprocessing as `bo_yield` — see `scripts/bo_yield/README.md`.

**NiB**: REACTANT and PRODUCT are canonicalized. REAGENT is combined from `CATALYST + REAGENT + SOLVENT` without pre-canonicalization, then canonicalized component-wise.

**SM / BH**: REACTANT, REAGENT, and PRODUCT are all canonicalized first, then REAGENT is combined from `CATALYST + REAGENT + SOLVENT` and canonicalized component-wise.

## Output

Output directory name: `gnn_bo_{total_trials}trials_conv{num_conv_layers}_{dataset}_seed{seed}`

```
runs/
└── gnn_bo_100trials_conv1_BH_seed1/
    ├── config.json
    ├── summary.json
    ├── optimization_log.csv
    └── visualization/
        ├── parity.png
        ├── error_hist.png
        ├── uncertainty_vs_yield.png
        ├── acquisition_vs_yield.png
        ├── optimization_progress.png
        ├── best_by_selection_type.png
        └── data_size_growth.png
```

**optimization_log.csv columns:**
`trial`, `index`, `selection_type`, `predicted_mean`, `predicted_std`, `actual_yield`,
`error_pct`, `acquisition_value`, `cumulative_data_size`, `best_so_far`

## Notes

- Graph data is constructed from SMILES on first run and cached for reuse
- The GNN is retrained from scratch at every iteration — GPU is strongly recommended
- `--num-conv-layers` controls model depth; shell scripts are provided for 1, 2, and 3 layers

## Dependencies

Managed via `pyproject.toml`. Key packages: `torch`, `torch-geometric`, `rdkit`, `scipy`, `matplotlib`

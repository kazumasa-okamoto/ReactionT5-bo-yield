# Gaussian Process Regression Bayesian Optimization

Scripts for Bayesian optimization using Gaussian Process Regression (GPR) with Morgan Fingerprints for reaction yield prediction.

> All commands are run from the **project root**.

## Files

- `data_utils.py`: Data loading, SMILES canonicalization, and Morgan Fingerprint computation
- `gpr_optimizer.py`: GPR Bayesian optimization (RBF kernel, Expected Improvement, sequential updates)
- `run_experiment.py`: Single experiment script
- `run_all_seeds_BH.sh`: Run BH experiments with seeds 1–5
- `run_all_seeds_NiB.sh`: Run NiB experiments with seeds 1–5
- `run_all_seeds_SM.sh`: Run SM experiments with seeds 1–5

## Usage

### Single Experiment

```bash
python scripts/gpr/run_experiment.py \
    --data data/NiB/inchi_23l_reaction_t5_ready.csv \
    --dataset-name NiB \
    --n-trials 100 \
    --seed 42
```

### Multiple Seeds

```bash
# Shell scripts (seeds 1–5)
bash scripts/gpr/run_all_seeds_BH.sh
bash scripts/gpr/run_all_seeds_NiB.sh
bash scripts/gpr/run_all_seeds_SM.sh
```

## Command Line Arguments

### run_experiment.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to CSV data file |
| `--dataset-name` | `"NiB"` | Dataset name (affects preprocessing) |
| `--n-trials` | `100` | Number of optimization trials |
| `--radius` | `4` | Morgan fingerprint radius |
| `--n-bits` | `2048` | Morgan fingerprint bits per molecule |
| `--output-dir` | `"runs"` | Base output directory |
| `--seed` | `42` | Random seed |

## Data Preprocessing

Same dataset-aware preprocessing as `bo_yield` — see `scripts/bo_yield/README.md`.

**NiB**: REACTANT and PRODUCT are canonicalized. REAGENT is combined from `CATALYST + REAGENT + SOLVENT` without pre-canonicalization, then canonicalized component-wise.

**SM / BH**: REACTANT, REAGENT, and PRODUCT are all canonicalized first, then REAGENT is combined from `CATALYST + REAGENT + SOLVENT` and canonicalized component-wise.

## Output

Output directory name: `gpr_{n_trials}trials_{dataset}_seed{seed}`

```
runs/
└── gpr_100trials_NiB_seed1/
    ├── config.json
    ├── summary.json
    └── optimization_log.csv
```

## Notes

- No model training required — fingerprints are computed once at the start
- GPR updates are sequential and fast (O(n³) in number of training samples)
- Uncertainty estimates are calibrated Bayesian posteriors

## Dependencies

Managed via `pyproject.toml`. Key packages: `scikit-learn`, `rdkit`, `scipy`, `numpy`, `pandas`

# Optuna TPE Bayesian Optimization

Scripts for Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE) for reaction yield prediction.

> All commands are run from the **project root**.

## Files

- `data_utils.py`: Data loading utilities (dataset-specific loaders for BH, NiB, SM)
- `tpe_optimizer.py`: TPE optimization core using Optuna
- `visualization.py`: Generate plots from Optuna study results
- `run_experiment_BH.py`: Single experiment script for Buchwald-Hartwig
- `run_experiment_NiB.py`: Single experiment script for NiB
- `run_experiment_SM.py`: Single experiment script for Suzuki-Miyaura
- `run_all_seeds_BH.sh`: Run BH experiments with seeds 1–5
- `run_all_seeds_NiB.sh`: Run NiB experiments with seeds 1–5
- `run_all_seeds_SM.sh`: Run SM experiments with seeds 1–5

## Usage

### Single Experiment

```bash
# Buchwald-Hartwig
python scripts/optuna_tpe/run_experiment_BH.py \
    --data data/Buchwald-Hartwig/Dreher_and_Doyle_reaction.csv \
    --n-trials 100 \
    --seed 42

# NiB
python scripts/optuna_tpe/run_experiment_NiB.py \
    --data data/NiB/inchi_23l.csv \
    --n-trials 100 \
    --seed 42

# Suzuki-Miyaura
python scripts/optuna_tpe/run_experiment_SM.py \
    --data data/Suzuki-Miyaura/aap9112_complete_grid.csv \
    --n-trials 100 \
    --seed 42
```

### Multiple Seeds

```bash
# Shell scripts (seeds 1–5)
bash scripts/optuna_tpe/run_all_seeds_BH.sh
bash scripts/optuna_tpe/run_all_seeds_NiB.sh
bash scripts/optuna_tpe/run_all_seeds_SM.sh
```

## Command Line Arguments

### run_experiment_{BH,NiB,SM}.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to CSV data file |
| `--n-trials` | `100` | Number of optimization trials |
| `--n-startup-trials` | `10` | Random trials before TPE starts |
| `--n-ei-candidates` | `24` | Candidate samples for EI computation |
| `--output-dir` | `"runs"` | Base output directory |
| `--seed` | `42` | Random seed |

## Data Preprocessing

Each dataset has a dedicated loader in `data_utils.py`. Reaction conditions (ligand, additive, base, aryl halide, etc.) are treated as categorical parameters for Optuna.

## Output

Output directory name: `optuna_tpe_{n_trials}trials_{dataset}_seed{seed}`

```
runs/
└── optuna_tpe_100trials_BH_seed1/
    ├── optuna_tpe_100trials_BH_seed1.db   # SQLite study database
    └── visualization/
        ├── optimization_progress.png
        ├── yield_distribution.png
        ├── top10_yields.png
        └── yield_statistics.png
```

Results can be loaded from the `.db` file using Optuna:

```python
import optuna
study = optuna.load_study(
    study_name="tpe_yield_optimization_BH",
    storage="sqlite:///runs/optuna_tpe_100trials_BH_seed1/optuna_tpe_100trials_BH_seed1.db"
)
print(f"Best yield: {study.best_value:.2f}%")
```

## Notes

- No model training required — TPE builds a surrogate from trial history
- Results are stored in SQLite; all trial history is preserved and queryable
- Each seed produces fully independent results

## Dependencies

Managed via `pyproject.toml`. Key packages: `optuna`, `pandas`, `matplotlib`, `numpy`

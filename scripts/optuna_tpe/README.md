# Optuna TPE Optimization Scripts

This directory contains scripts for running Optuna TPE (Tree-structured Parzen Estimator) optimization experiments on reaction yield datasets.

## Overview

The scripts implement TPE-based Bayesian optimization for maximizing reaction yields across three datasets:
- **Buchwald-Hartwig (BH)**: Cross-coupling reactions
- **NiB**: Nickel-catalyzed borylation reactions
- **Suzuki-Miyaura (SM)**: Palladium-catalyzed cross-coupling reactions

## Directory Structure

```
optuna_tpe/
├── data_utils.py              # Data loading utilities
├── tpe_optimizer.py           # TPE optimization core functionality
├── visualization.py           # Visualization utilities for results
├── run_experiment_BH.py       # BH dataset experiment script
├── run_experiment_NiB.py      # NiB dataset experiment script
├── run_experiment_SM.py       # SM dataset experiment script
├── run_all_seeds_BH.sh        # Run BH experiments with multiple seeds
├── run_all_seeds_NiB.sh       # Run NiB experiments with multiple seeds
├── run_all_seeds_SM.sh        # Run SM experiments with multiple seeds
└── README.md                  # This file
```

## Requirements

The scripts require the following Python packages:
- pandas
- optuna
- matplotlib
- numpy

These are already included in the project's `pyproject.toml` dependencies.

## Usage

### Running All Seeds for a Dataset

The easiest way to run experiments is using the shell scripts that automatically run multiple seeds (1, 2, 3, 4, 5):

```bash
cd scripts/optuna_tpe

# Buchwald-Hartwig dataset
./run_all_seeds_BH.sh

# NiB dataset
./run_all_seeds_NiB.sh

# Suzuki-Miyaura dataset
./run_all_seeds_SM.sh
```

### Running Individual Experiments

You can also run individual experiments with custom parameters:

#### Buchwald-Hartwig Dataset
```bash
python run_experiment_BH.py \
    --data ../../data/Buchwald-Hartwig/Dreher_and_Doyle_reaction.csv \
    --n-trials 100 \
    --seed 42 \
    --output-dir ../../runs \
    --n-startup-trials 10 \
    --n-ei-candidates 24
```

#### NiB Dataset
```bash
python run_experiment_NiB.py \
    --data ../../data/NiB/inchi_23l.csv \
    --n-trials 100 \
    --seed 42 \
    --output-dir ../../runs \
    --n-startup-trials 10 \
    --n-ei-candidates 24
```

#### Suzuki-Miyaura Dataset
```bash
python run_experiment_SM.py \
    --data ../../data/Suzuki-Miyaura/aap9112_complete_grid.csv \
    --n-trials 100 \
    --seed 42 \
    --output-dir ../../runs \
    --n-startup-trials 10 \
    --n-ei-candidates 24
```

## Command Line Arguments

All experiment scripts support the following arguments:

- `--data`: Path to the CSV data file (required)
- `--n-trials`: Number of optimization trials (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Output directory for results (default: ../../runs)
- `--n-startup-trials`: Number of random sampling trials before TPE starts (default: 10)
- `--n-ei-candidates`: Number of candidate samples for expected improvement (default: 24)

## Output

Each experiment creates an output directory with the following structure:

```
runs/
└── optuna_tpe_{n_trials}trials_{dataset}_seed{seed}/
    ├── optuna_tpe_{n_trials}trials_{dataset}_seed{seed}.db
    └── visualization/
        ├── optimization_progress.png
        ├── yield_distribution.png
        ├── top10_yields.png
        └── yield_statistics.png
```

The `.db` file is a SQLite database containing:
- All trial results
- Best parameters found
- Optimization history
- User attributes for each trial

### Visualization Output

Each experiment automatically generates four visualization plots in the `visualization/` subdirectory:

1. **optimization_progress.png**: Shows the actual yield at each trial and the best yield found so far
2. **yield_distribution.png**: Histogram of yield values explored during optimization
3. **top10_yields.png**: Bar chart of the top 10 highest yields discovered
4. **yield_statistics.png**: Statistical summary (min, Q1, median, Q3, max) of yields

## TPE Sampler Configuration

The TPE sampler is configured with:
- **n_startup_trials**: 10 (initial random exploration trials)
- **n_ei_candidates**: 24 (number of candidates for expected improvement)
- **seed**: Configurable (1, 2, 3, 4, 5 in the shell scripts)

These parameters control the exploration-exploitation trade-off in the optimization process.

## Loading Results

You can load and analyze results using Optuna:

```python
import optuna

# Load study
study = optuna.load_study(
    study_name='tpe_yield_optimization_BH',  # or _NiB, _SM
    storage='sqlite:///path/to/experiment.db'
)

# Best result
print(f"Best yield: {study.best_value:.2f}%")
print(f"Best parameters: {study.best_params}")

# All trials
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value:.2f}%")
```

## Comparison with Notebooks

These scripts are based on the Optuna notebooks:
- `notebooks/optuna_yield_BH.ipynb`
- `notebooks/optuna_yield_NiB.ipynb`
- `notebooks/optuna_yield_SM.ipynb`

The main differences:
- Scripts are designed for batch execution with multiple seeds
- Results are saved to SQLite databases for later analysis
- Progress bars show optimization status
- Command-line interface for easy parameter configuration
- Automatic visualization generation after each experiment

## Notes

- Each seed produces independent optimization results
- Running all 5 seeds allows for statistical analysis of optimization performance
- The optimization is deterministic given a fixed seed
- Results can be compared with other methods (GPR, Bayesian optimization with fine-tuning)

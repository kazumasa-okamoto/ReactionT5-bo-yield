# GPR Bayesian Optimization Experiment Scripts

This directory (`scripts/gpr/`) contains scripts for running Gaussian Process Regression (GPR) based Bayesian optimization experiments for reaction yield prediction.

## Location

All scripts are located in `scripts/gpr/` directory. Run commands from this directory:
```bash
cd scripts/gpr
```

## Files

- `data_utils.py`: Data processing and Morgan Fingerprint computation
  - **Dataset-aware preprocessing**: NiB vs other datasets (SM, BH)
  - SMILES canonicalization with RDKit
  - Morgan Fingerprint generation (radius=2, n_bits=2048)
  - Reaction dictionary creation
- `gpr_optimizer.py`: Gaussian Process Regression Bayesian Optimization
  - Scikit-learn GPR with RBF kernel
  - Expected Improvement acquisition function
  - Sequential model updates
- `run_experiment.py`: Main experiment script
- `run_all_seeds_NiB.sh`: Run NiB experiments with seeds 1, 2, 3, 4, 5
- `run_all_seeds_SM.sh`: Run SM experiments with seeds 1, 2, 3, 4, 5
- `run_all_seeds_BH.sh`: Run BH experiments with seeds 1, 2, 3, 4, 5

## Features

- **Morgan Fingerprint**: Uses Morgan Fingerprint (ECFP) as molecular features
- **Gaussian Process Regression**: Non-parametric Bayesian method with uncertainty quantification
- **Expected Improvement**: Standard acquisition function for Bayesian optimization
- **No Fine-tuning**: Uses pre-computed fingerprints, no model training required
- **Sequential Updates**: GPR model updated after each trial

## Usage

### Single Experiment

Run a single experiment with a specific seed:

```bash
cd scripts/gpr
python run_experiment.py \
    --data ../../data/NiB/inchi_23l_reaction_t5_ready.csv \
    --dataset-name NiB \
    --n-trials 100 \
    --seed 42
```

### Multiple Seeds

Run experiments with multiple seeds (1, 2, 3, 4, 5):

**NiB Dataset:**
```bash
cd scripts/gpr
bash run_all_seeds_NiB.sh
```

**SM Dataset (Suzuki-Miyaura):**
```bash
cd scripts/gpr
bash run_all_seeds_SM.sh
```

**BH Dataset (Buchwald-Hartwig):**
```bash
cd scripts/gpr
bash run_all_seeds_BH.sh
```

## Command Line Arguments

### run_experiment.py

**Data arguments:**
- `--data`: Path to CSV data file (required)
- `--dataset-name`: Dataset name (default: "NiB")

**GPR arguments:**
- `--n-trials`: Number of optimization trials (default: 100)
- `--radius`: Morgan fingerprint radius (default: 2)
- `--n-bits`: Morgan fingerprint n_bits per molecule (default: 2048)

**Output arguments:**
- `--output-dir`: Base output directory (default: "runs")

**Reproducibility:**
- `--seed`: Random seed (default: 42)

## Data Preprocessing

Same as bo_yield - see `scripts/bo_yield/README.md` for details.

### NiB Dataset
- **REACTANT**: Canonicalized and sorted
- **PRODUCT**: Canonicalized and sorted
- **REAGENT**: Combined from `CATALYST + REAGENT + SOLVENT` (NOT canonicalized)

### Other Datasets (SM, BH)
- **REACTANT**: Canonicalized and sorted
- **REAGENT**: Canonicalized and sorted
- **PRODUCT**: Canonicalized and sorted

## Output Structure

```
ReactionT5-bo-yield/
|-- scripts/
|   `-- gpr/                        # GPR scripts (this directory)
|       |-- run_experiment.py
|       |-- run_all_seeds_NiB.sh
|       `-- ...
`-- runs/                           # Experiment results
    |-- gpr_100trials_NiB_seed1/
    |   |-- config.json             # Experiment configuration
    |   |-- summary.json            # Summary statistics
    |   `-- optimization_log.csv    # Detailed optimization log
    |-- gpr_100trials_NiB_seed2/
    `-- ...
```

## Differences from bo_yield

| Feature | GPR (this) | Bayesian Optimization (bo_yield) |
|---------|------------|----------------------------------|
| Model | Gaussian Process Regression | ReactionT5 + MC Dropout |
| Features | Morgan Fingerprint (6144-dim) | Reaction SMILES (text) |
| Uncertainty | GPR posterior variance | MC Dropout variance |
| Training | Sequential GPR updates | Fine-tuning per round |
| Speed | Fast (fingerprint-based) | Slower (model inference) |
| Interpretability | High (feature-based) | Lower (end-to-end) |

## Example: Running All Datasets

```bash
cd scripts/gpr

# Run all three datasets in parallel (if resources allow)
bash run_all_seeds_NiB.sh &
bash run_all_seeds_SM.sh &
bash run_all_seeds_BH.sh &

# Or run sequentially
bash run_all_seeds_NiB.sh
bash run_all_seeds_SM.sh
bash run_all_seeds_BH.sh
```

## Notes

- GPR is computationally efficient compared to deep learning methods
- Fingerprint computation is done once at the beginning
- Model updates are fast (O(n^3) for n training samples)
- Works well for small to medium datasets (< 10,000 samples)
- Uncertainty estimates are calibrated (proper Bayesian posterior)

## Dependencies

Key dependencies (in addition to bo_yield dependencies):
- scikit-learn >= 1.0
- rdkit
- scipy
- numpy
- pandas

All dependencies are managed via pyproject.toml.

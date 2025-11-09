"""Optuna TPE optimization for Buchwald-Hartwig dataset."""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_buchwald_hartwig_data
from tpe_optimizer import run_tpe_optimization, print_optimization_results
from visualization import create_visualizations, print_statistics


def create_objective_function(yield_dict, ligand_list, additive_list, base_list, aryl_halide_list):
    """
    Create objective function for Buchwald-Hartwig optimization.

    Args:
        yield_dict: Dictionary mapping reaction conditions to yields
        ligand_list: List of ligand options
        additive_list: List of additive options
        base_list: List of base options
        aryl_halide_list: List of aryl halide options

    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Sample categorical parameters
        ligand = trial.suggest_categorical('ligand', ligand_list)
        additive = trial.suggest_categorical('additive', additive_list)
        base = trial.suggest_categorical('base', base_list)
        aryl_halide = trial.suggest_categorical('aryl_halide', aryl_halide_list)

        # Create search key
        key = (ligand, additive, base, aryl_halide)

        # Get yield from experimental data
        if key in yield_dict:
            yield_value = yield_dict[key]
        else:
            yield_value = 0.0

        # Save additional information for later analysis
        trial.set_user_attr('ligand', ligand)
        trial.set_user_attr('additive', additive)
        trial.set_user_attr('base', base)
        trial.set_user_attr('aryl_halide', aryl_halide)
        trial.set_user_attr('yield', yield_value)

        return yield_value

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna TPE optimization for Buchwald-Hartwig dataset"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV data file")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="../../runs",
                        help="Output directory for results")
    parser.add_argument("--n-startup-trials", type=int, default=10,
                        help="Number of random sampling trials before TPE starts")
    parser.add_argument("--n-ei-candidates", type=int, default=24,
                        help="Number of candidate samples for expected improvement")

    args = parser.parse_args()

    # Create experiment name and output paths
    exp_name = f"optuna_tpe_{args.n_trials}trials_BH_seed{args.seed}"
    output_path = os.path.join(args.output_dir, exp_name)
    db_path = os.path.join(output_path, f"{exp_name}.db")

    print("=" * 80)
    print("OPTUNA TPE OPTIMIZATION - BUCHWALD-HARTWIG DATASET")
    print("=" * 80)
    print(f"Data path: {args.data}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_path}")
    print(f"Database path: {db_path}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    yield_dict, ligand_list, additive_list, base_list, aryl_halide_list = load_buchwald_hartwig_data(args.data)

    # Create objective function
    objective = create_objective_function(
        yield_dict, ligand_list, additive_list, base_list, aryl_halide_list
    )

    # Run optimization
    print("\nStarting TPE optimization...")
    study = run_tpe_optimization(
        objective_fn=objective,
        study_name='tpe_yield_optimization_BH',
        db_path=db_path,
        n_trials=args.n_trials,
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=args.n_ei_candidates,
        show_progress_bar=True
    )

    # Print results
    print_optimization_results(study, db_path)

    # Print detailed statistics
    print_statistics(study)

    # Create visualizations
    viz_dir = create_visualizations(study, output_path, dataset_name="Buchwald-Hartwig")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results directory: {output_path}")
    print(f"Database: {db_path}")
    print(f"Visualizations: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

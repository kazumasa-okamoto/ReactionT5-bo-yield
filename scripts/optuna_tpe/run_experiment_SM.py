"""Optuna TPE optimization for Suzuki-Miyaura dataset."""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_suzuki_miyaura_data
from tpe_optimizer import run_tpe_optimization, print_optimization_results
from visualization import create_visualizations, print_statistics


def create_objective_function(yield_dict, reactant_1_list, reactant_2_list,
                               catalyst_list, ligand_list, reagent_list, solvent_list):
    """
    Create objective function for Suzuki-Miyaura optimization.

    Args:
        yield_dict: Dictionary mapping reaction conditions to yields
        reactant_1_list: List of reactant 1 options
        reactant_2_list: List of reactant 2 options
        catalyst_list: List of catalyst options
        ligand_list: List of ligand options
        reagent_list: List of reagent options
        solvent_list: List of solvent options

    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Sample categorical parameters
        reactant_1_name = trial.suggest_categorical('reactant_1_name', reactant_1_list)
        reactant_2_name = trial.suggest_categorical('reactant_2_name', reactant_2_list)
        catalyst_1_short_hand = trial.suggest_categorical('catalyst_1_short_hand', catalyst_list)
        ligand_short_hand = trial.suggest_categorical('ligand_short_hand', ligand_list)
        reagent_1_short_hand = trial.suggest_categorical('reagent_1_short_hand', reagent_list)
        solvent_1_short_hand = trial.suggest_categorical('solvent_1_short_hand', solvent_list)

        # Create search key
        key = (reactant_1_name, reactant_2_name, catalyst_1_short_hand,
               ligand_short_hand, reagent_1_short_hand, solvent_1_short_hand)

        # Get yield from experimental data
        if key in yield_dict:
            yield_value = yield_dict[key]
        else:
            yield_value = 0.0

        # Save additional information for later analysis
        trial.set_user_attr('reactant_1_name', reactant_1_name)
        trial.set_user_attr('reactant_2_name', reactant_2_name)
        trial.set_user_attr('catalyst_1_short_hand', catalyst_1_short_hand)
        trial.set_user_attr('ligand_short_hand', ligand_short_hand)
        trial.set_user_attr('reagent_1_short_hand', reagent_1_short_hand)
        trial.set_user_attr('solvent_1_short_hand', solvent_1_short_hand)
        trial.set_user_attr('yield', yield_value)

        return yield_value

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna TPE optimization for Suzuki-Miyaura dataset"
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
    exp_name = f"optuna_tpe_{args.n_trials}trials_SM_seed{args.seed}"
    output_path = os.path.join(args.output_dir, exp_name)
    db_path = os.path.join(output_path, f"{exp_name}.db")

    print("=" * 80)
    print("OPTUNA TPE OPTIMIZATION - SUZUKI-MIYAURA DATASET")
    print("=" * 80)
    print(f"Data path: {args.data}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_path}")
    print(f"Database path: {db_path}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    (yield_dict, reactant_1_list, reactant_2_list,
     catalyst_list, ligand_list, reagent_list, solvent_list) = load_suzuki_miyaura_data(args.data)

    # Create objective function
    objective = create_objective_function(
        yield_dict, reactant_1_list, reactant_2_list,
        catalyst_list, ligand_list, reagent_list, solvent_list
    )

    # Run optimization
    print("\nStarting TPE optimization...")
    study = run_tpe_optimization(
        objective_fn=objective,
        study_name='tpe_yield_optimization_SM',
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
    viz_dir = create_visualizations(study, output_path, dataset_name="Suzuki-Miyaura")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results directory: {output_path}")
    print(f"Database: {db_path}")
    print(f"Visualizations: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

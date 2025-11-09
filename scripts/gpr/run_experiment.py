"""Main experiment script for GPR Bayesian optimization."""

import os
import argparse
import json
import random
import numpy as np

from data_utils import (
    load_and_preprocess_data,
    compute_all_fingerprints,
    create_reaction_dictionaries
)
from gpr_optimizer import GaussianProcessBayesianOptimization, GPRConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Run GPR Bayesian optimization experiment")

    # Data arguments
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--dataset-name", type=str, default="NiB", help="Dataset name")

    # GPR arguments
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--radius", type=int, default=4, help="Morgan fingerprint radius")
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint n_bits")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Create output directory
    exp_name = f"gpr_{args.n_trials}trials_{args.dataset_name}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config_dict = vars(args)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {output_dir}/config.json")

    # Load and preprocess data
    print("\n" + "=" * 80)
    print("Loading and preprocessing data...")
    print("=" * 80)
    df = load_and_preprocess_data(args.data, dataset_name=args.dataset_name)
    print(f"Loaded {len(df)} reactions")
    print(f"Dataset type: {args.dataset_name}")

    # Compute fingerprints
    print("\n" + "=" * 80)
    print("Computing Morgan Fingerprints...")
    print("=" * 80)
    X, y = compute_all_fingerprints(df, radius=args.radius, n_bits=args.n_bits)

    # Create reaction dictionaries
    print("\n" + "=" * 80)
    print("Creating reaction dictionaries...")
    print("=" * 80)
    reactant_list, reagent_list, product_list, product_dict, true_yield_dict, fingerprint_dict = \
        create_reaction_dictionaries(df, X)

    # Create configuration
    config = GPRConfig(
        n_trials=args.n_trials,
        radius=args.radius,
        n_bits=args.n_bits,
        output_dir=output_dir,
        study_seed=args.seed
    )

    # Initialize optimizer
    print("\n" + "=" * 80)
    print("Initializing GPR optimizer...")
    print("=" * 80)
    optimizer = GaussianProcessBayesianOptimization(
        reactant_list=reactant_list,
        reagent_list=reagent_list,
        product_dict=product_dict,
        true_yield_dict=true_yield_dict,
        fingerprint_dict=fingerprint_dict,
        config=config
    )

    # Run optimization
    print("\n" + "=" * 80)
    print("Starting optimization...")
    print("=" * 80)
    best_result = optimizer.optimize()

    # Get summary
    summary = optimizer.get_optimization_summary()

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_dir}/summary.json")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)
    optimizer.save_visualization()

    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total trials: {summary.get('total_trials', 0)}")
    print(f"Best yield: {summary.get('max_yield', 0):.2f}%")
    print(f"Mean yield: {summary.get('mean_yield', 0):.2f}%")
    print(f"Std yield: {summary.get('std_yield', 0):.2f}%")
    if summary.get('mae_error'):
        print(f"MAE: {summary['mae_error']:.2f}%")
        print(f"RMSE: {summary['rmse_error']:.2f}%")
    print(f"Coverage: {summary.get('coverage', 0):.1f}%")
    print(f"Log file: {summary.get('log_path', 'N/A')}")

    if best_result:
        print("\n" + "=" * 80)
        print("BEST COMBINATION")
        print("=" * 80)
        print(f"Yield: {best_result['actual_yield']:.2f}%")
        print(f"Reactant: {best_result['reactant']}")
        print(f"Reagent: {best_result['reagent']}")
        print(f"Product: {best_result['product']}")
        print(f"Found in: Trial {best_result['trial']}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

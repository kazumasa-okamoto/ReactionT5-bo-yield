"""Main experiment script for Bayesian optimization with fine-tuning."""

import os
import argparse
import json
import random
import numpy as np
import torch

from data_utils import load_and_preprocess_data, create_reaction_dictionaries
from model_utils import load_model_and_tokenizer
from bayesian_optimizer import BayesianOptimizationWithFineTuning, LoopConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian optimization experiment for reaction yield")

    # Data arguments
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--dataset-name", type=str, default="NiB", help="Dataset name (for output directory)")

    # Optimization arguments
    parser.add_argument("--n-rounds", type=int, default=10, help="Number of optimization rounds")
    parser.add_argument("--trials-per-round", type=int, default=10, help="Number of trials per round")
    parser.add_argument("--n-mc-samples", type=int, default=10, help="Number of MC Dropout samples")
    parser.add_argument("--batch-size-prediction", type=int, default=64, help="Batch size for prediction")

    # Fine-tuning arguments
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--epochs-per-round", type=int, default=2, help="Training epochs per round")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size-train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch-size-eval", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="sagawa/ReactionT5v2-yield",
                        help="Model name or path")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="runs", help="Base output directory")
    parser.add_argument("--no-checkpoints", action="store_true", help="Do not save model checkpoints")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Create output directory
    exp_name = f"bayes_{args.n_rounds}rounds_{args.trials_per_round}trials_{args.dataset_name}_seed{args.seed}"
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

    # Create reaction dictionaries
    reactant_list, reagent_list, product_list, product_dict, true_yield_dict = create_reaction_dictionaries(df)

    # Load model and tokenizer
    print("\n" + "=" * 80)
    print("Loading model and tokenizer...")
    print("=" * 80)
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print(f"Model loaded: {args.model_name}")

    # Create configuration
    config = LoopConfig(
        n_rounds=args.n_rounds,
        trials_per_round=args.trials_per_round,
        n_mc_samples=args.n_mc_samples,
        batch_size_prediction=args.batch_size_prediction,
        learning_rate=args.learning_rate,
        epochs_per_round=args.epochs_per_round,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        batch_size_train=args.batch_size_train,
        batch_size_eval=args.batch_size_eval,
        val_ratio=args.val_ratio,
        output_dir=output_dir,
        save_checkpoints=not args.no_checkpoints,
        study_seed=args.seed
    )

    # Initialize optimizer
    print("\n" + "=" * 80)
    print("Initializing Bayesian optimizer...")
    print("=" * 80)
    optimizer = BayesianOptimizationWithFineTuning(
        model=model,
        tokenizer=tokenizer,
        reactant_list=reactant_list,
        reagent_list=reagent_list,
        product_dict=product_dict,
        true_yield_dict=true_yield_dict,
        config=config
    )

    # Run optimization
    print("\n" + "=" * 80)
    print("Starting optimization...")
    print("=" * 80)
    best_result = optimizer.optimize_with_finetuning()

    # Get summary
    summary = optimizer.get_optimization_summary()

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_dir}/summary.json")

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
    print(f"Training data size: {summary.get('training_data_size', 0)}")
    print(f"Log file: {summary.get('log_path', 'N/A')}")

    if best_result:
        print("\n" + "=" * 80)
        print("BEST COMBINATION")
        print("=" * 80)
        print(f"Yield: {best_result['actual_yield']:.2f}%")
        print(f"Reactant: {best_result['reactant']}")
        print(f"Reagent: {best_result['reagent']}")
        print(f"Product: {best_result['product']}")
        print(f"Found in: Round {best_result['round']}, Trial {best_result['trial']}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

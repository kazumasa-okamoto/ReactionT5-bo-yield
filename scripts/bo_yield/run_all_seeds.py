"""Run experiments with multiple seeds (Python version)."""

import os
import subprocess
import argparse
from pathlib import Path


def run_experiment(
    data_path: str,
    dataset_name: str,
    seed: int,
    n_rounds: int = 10,
    trials_per_round: int = 10,
    n_mc_samples: int = 10,
    batch_size_prediction: int = 64,
    learning_rate: float = 5e-4,
    epochs_per_round: int = 2,
    weight_decay: float = 0.01,
    batch_size_train: int = 8,
    batch_size_eval: int = 16,
    val_ratio: float = 0.2,
    output_dir: str = "../runs"
):
    """Run a single experiment with specified seed."""

    cmd = [
        "python", "run_experiment.py",
        "--data", data_path,
        "--dataset-name", dataset_name,
        "--n-rounds", str(n_rounds),
        "--trials-per-round", str(trials_per_round),
        "--n-mc-samples", str(n_mc_samples),
        "--batch-size-prediction", str(batch_size_prediction),
        "--learning-rate", str(learning_rate),
        "--epochs-per-round", str(epochs_per_round),
        "--weight-decay", str(weight_decay),
        "--batch-size-train", str(batch_size_train),
        "--batch-size-eval", str(batch_size_eval),
        "--val-ratio", str(val_ratio),
        "--output-dir", output_dir,
        "--seed", str(seed)
    ]

    print(f"\n{'=' * 80}")
    print(f"Starting experiment with seed: {seed}")
    print(f"{'=' * 80}")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[Success] Experiment with seed {seed} completed successfully")
        return True
    else:
        print(f"[Failure] Experiment with seed {seed} failed")
        return False


def visualize_experiment(
    output_dir: str,
    dataset_name: str,
    seed: int,
    n_rounds: int = 10,
    trials_per_round: int = 10
):
    """Generate visualizations for an experiment."""

    exp_name = f"bayes_{n_rounds}rounds_{trials_per_round}trials_{dataset_name}_seed{seed}"
    csv_path = os.path.join(output_dir, exp_name, "optimization_log.csv")
    logdir = os.path.join(output_dir, exp_name)

    if not os.path.exists(csv_path):
        print(f"[Warning] CSV file not found for seed {seed}: {csv_path}")
        return False

    print(f"Visualizing seed {seed}...")

    cmd = [
        "python", "visualize_results.py",
        "--csv", csv_path,
        "--logdir", logdir
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"[Success] Visualization for seed {seed} completed")
        return True
    else:
        print(f"[Failure] Visualization for seed {seed} failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experiments with multiple seeds")

    # Data arguments
    parser.add_argument("--data", type=str, default="../../data/NiB/inchi_23l_reaction_t5_ready.csv",
                        help="Path to CSV data file")
    parser.add_argument("--dataset-name", type=str, default="NiB", help="Dataset name")

    # Optimization arguments
    parser.add_argument("--n-rounds", type=int, default=10, help="Number of optimization rounds")
    parser.add_argument("--trials-per-round", type=int, default=10, help="Number of trials per round")
    parser.add_argument("--n-mc-samples", type=int, default=10, help="Number of MC Dropout samples")
    parser.add_argument("--batch-size-prediction", type=int, default=64, help="Batch size for prediction")

    # Fine-tuning arguments
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs-per-round", type=int, default=2, help="Training epochs per round")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size-train", type=int, default=8, help="Batch size for training")
    parser.add_argument("--batch-size-eval", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="../../runs", help="Base output directory")

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of random seeds to run")

    # Control flags
    parser.add_argument("--skip-visualization", action="store_true",
                        help="Skip visualization step")

    args = parser.parse_args()

    print(f"{'=' * 80}")
    print("Running experiments with multiple seeds")
    print(f"{'=' * 80}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Data path: {args.data}")
    print(f"N rounds: {args.n_rounds}")
    print(f"Trials per round: {args.trials_per_round}")
    print(f"Seeds: {args.seeds}")
    print(f"{'=' * 80}")

    # Run experiments
    success_count = 0
    failed_seeds = []

    for seed in args.seeds:
        success = run_experiment(
            data_path=args.data,
            dataset_name=args.dataset_name,
            seed=seed,
            n_rounds=args.n_rounds,
            trials_per_round=args.trials_per_round,
            n_mc_samples=args.n_mc_samples,
            batch_size_prediction=args.batch_size_prediction,
            learning_rate=args.learning_rate,
            epochs_per_round=args.epochs_per_round,
            weight_decay=args.weight_decay,
            batch_size_train=args.batch_size_train,
            batch_size_eval=args.batch_size_eval,
            val_ratio=args.val_ratio,
            output_dir=args.output_dir
        )

        if success:
            success_count += 1
        else:
            failed_seeds.append(seed)

    print(f"\n{'=' * 80}")
    print(f"All experiments completed!")
    print(f"{'=' * 80}")
    print(f"Successful: {success_count}/{len(args.seeds)}")
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")
    print(f"Results saved to: {args.output_dir}")

    # Run visualizations
    if not args.skip_visualization and success_count > 0:
        print(f"\n{'=' * 80}")
        print("Generating visualizations...")
        print(f"{'=' * 80}")

        for seed in args.seeds:
            if seed not in failed_seeds:
                visualize_experiment(
                    output_dir=args.output_dir,
                    dataset_name=args.dataset_name,
                    seed=seed,
                    n_rounds=args.n_rounds,
                    trials_per_round=args.trials_per_round
                )

        print(f"\n{'=' * 80}")
        print("All visualizations completed!")
        print(f"{'=' * 80}")

    # Exit with error if any experiment failed
    if failed_seeds:
        exit(1)


if __name__ == "__main__":
    main()

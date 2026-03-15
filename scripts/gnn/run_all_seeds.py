"""Run GNN experiments with multiple seeds."""

import os
import subprocess
import argparse


def run_experiment(
    data_path: str,
    dataset_name: str,
    seed: int,
    n_initial_random: int = 10,
    n_bo_iterations: int = 90,
    n_mc_samples: int = 10,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    early_stopping_patience: int = 10,
    hidden_dim: int = 256,
    dropout_rate: float = 0.2,
    num_conv_layers: int = 1,
    output_dir: str = "../../runs",
    save_checkpoints: bool = False,
    graph_data_dir: str = None
):
    """Run a single experiment with specified seed."""

    cmd = [
        "python", "run_experiment.py",
        "--data", data_path,
        "--dataset-name", dataset_name,
        "--n-initial-random", str(n_initial_random),
        "--n-bo-iterations", str(n_bo_iterations),
        "--n-mc-samples", str(n_mc_samples),
        "--learning-rate", str(learning_rate),
        "--num-epochs", str(num_epochs),
        "--weight-decay", str(weight_decay),
        "--batch-size", str(batch_size),
        "--val-ratio", str(val_ratio),
        "--early-stopping-patience", str(early_stopping_patience),
        "--hidden-dim", str(hidden_dim),
        "--dropout-rate", str(dropout_rate),
        "--num-conv-layers", str(num_conv_layers),
        "--output-dir", output_dir,
        "--seed", str(seed)
    ]

    if save_checkpoints:
        cmd.append("--save-checkpoints")

    if graph_data_dir:
        cmd.extend(["--graph-data-dir", graph_data_dir])

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


def main():
    parser = argparse.ArgumentParser(description="Run GNN experiments with multiple seeds")

    # Data arguments
    parser.add_argument("--data", type=str, default="../../data/Buchwald-Hartwig/Dreher_and_Doyle_reaction_t5_ready.csv",
                        help="Path to CSV data file")
    parser.add_argument("--dataset-name", type=str, default="BH", help="Dataset name")
    parser.add_argument("--graph-data-dir", type=str, default=None, help="Directory to save/load graph data")

    # Optimization arguments
    parser.add_argument("--n-initial-random", type=int, default=10, help="Number of initial random samples")
    parser.add_argument("--n-bo-iterations", type=int, default=90, help="Number of BO iterations")
    parser.add_argument("--n-mc-samples", type=int, default=10, help="Number of MC Dropout samples")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Training epochs per iteration")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience (0 to disable)")

    # Model arguments
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout-rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num-conv-layers", type=int, default=1, choices=[1, 2, 3],
                        help="Number of graph convolutional layers (1, 2, or 3)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="../../runs", help="Base output directory")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model checkpoints")

    # Seeds
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of random seeds to run")

    args = parser.parse_args()

    print(f"{'=' * 80}")
    print("Running GNN experiments with multiple seeds")
    print(f"{'=' * 80}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Data path: {args.data}")
    print(f"Initial random samples: {args.n_initial_random}")
    print(f"BO iterations: {args.n_bo_iterations}")
    print(f"Total trials: {args.n_initial_random + args.n_bo_iterations}")
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
            n_initial_random=args.n_initial_random,
            n_bo_iterations=args.n_bo_iterations,
            n_mc_samples=args.n_mc_samples,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            early_stopping_patience=args.early_stopping_patience,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            num_conv_layers=args.num_conv_layers,
            output_dir=args.output_dir,
            save_checkpoints=args.save_checkpoints,
            graph_data_dir=args.graph_data_dir
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

    # Exit with error if any experiment failed
    if failed_seeds:
        exit(1)


if __name__ == "__main__":
    main()

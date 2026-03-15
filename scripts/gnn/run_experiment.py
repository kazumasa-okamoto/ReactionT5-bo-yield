"""Main experiment script for GNN-based Bayesian optimization."""

import os
import argparse
import json
import random
import numpy as np
import torch

from data_utils import load_and_preprocess_data, process_and_save_graphs, ReactionDataset
from bayesian_optimizer import BayesianOptimizationGNN, LoopConfig


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
    parser = argparse.ArgumentParser(description="Run GNN Bayesian optimization experiment")

    # Data arguments
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
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

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Create output directory
    exp_name = f"gnn_bo_{args.n_initial_random + args.n_bo_iterations}trials_conv{args.num_conv_layers}_{args.dataset_name}_seed{args.seed}"
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

    # Process and save graphs
    if args.graph_data_dir is None:
        graph_data_dir = os.path.join(os.path.dirname(args.data), f"{args.dataset_name}_graph_data")
    else:
        graph_data_dir = args.graph_data_dir

    # Check if graph data already exists
    if not os.path.exists(graph_data_dir) or len(os.listdir(graph_data_dir)) == 0:
        print("\n" + "=" * 80)
        print("Processing reactions to graphs...")
        print("=" * 80)
        symbols, total_count = process_and_save_graphs(df, graph_data_dir)
    else:
        print(f"\nGraph data already exists at: {graph_data_dir}")
        symbols = torch.load(os.path.join(graph_data_dir, 'symbols.pt'))
        print(f"Loaded symbols: {symbols}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading graph dataset...")
    print("=" * 80)
    reaction_dataset = ReactionDataset(root=graph_data_dir)

    # Calculate node feature dimension
    node_in_dim = len(symbols) + 3  # atomic symbols + degree + valence + aromatic
    print(f"Node input dimension: {node_in_dim}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create configuration
    config = LoopConfig(
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
        output_dir=output_dir,
        save_checkpoints=args.save_checkpoints,
        seed=args.seed
    )

    # Initialize optimizer
    print("\n" + "=" * 80)
    print("Initializing Bayesian optimizer...")
    print("=" * 80)
    optimizer = BayesianOptimizationGNN(
        dataset=reaction_dataset,
        node_in_dim=node_in_dim,
        device=device,
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

    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total trials: {summary.get('total_trials', 0)}")
    print(f"Random trials: {summary.get('random_trials', 0)}")
    print(f"Bayesian trials: {summary.get('bayesian_trials', 0)}")
    print(f"Best yield: {summary.get('max_yield', 0):.2f}%")
    print(f"Mean yield: {summary.get('mean_yield', 0):.2f}%")
    print(f"Std yield: {summary.get('std_yield', 0):.2f}%")
    if summary.get('mae_error'):
        print(f"MAE: {summary['mae_error']:.2f}%")
        print(f"RMSE: {summary['rmse_error']:.2f}%")
    print(f"Coverage: {summary.get('coverage', 0):.2f}%")
    print(f"Training data size: {summary.get('training_data_size', 0)}")
    print(f"Log file: {summary.get('log_path', 'N/A')}")

    if best_result:
        print("\n" + "=" * 80)
        print("BEST RESULT")
        print("=" * 80)
        print(f"Yield: {best_result['actual_yield']:.2f}%")
        print(f"Index: {best_result['index']}")
        print(f"Found at iteration: {best_result['iteration']}")
        print(f"Selection method: {best_result['selection_type']}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

"""Visualization utilities for Optuna TPE optimization results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import optuna
from typing import Optional


def create_visualizations(study: optuna.Study, output_dir: str, dataset_name: str = ""):
    """
    Create and save visualization plots for optimization results.

    Args:
        study: Completed optuna.Study object
        output_dir: Directory to save visualization plots
        dataset_name: Name of dataset for plot titles
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    # Extract trial values
    trial_values = [t.value for t in study.trials if t.value is not None]
    best_values = np.maximum.accumulate(trial_values)

    print(f"\nCreating visualizations...")
    print(f"Visualization directory: {viz_dir}")

    # Set plot style
    plt.style.use('default')

    # 1. Optimization Progress
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(trial_values)+1), trial_values, 'o-', alpha=0.6,
             label='Actual Yield', markersize=4)
    plt.plot(range(1, len(trial_values)+1), best_values, 'r-', linewidth=3,
             label='Best So Far')
    plt.xlabel('Trial')
    plt.ylabel('Yield [%]')
    title = 'TPE Optimization Progress'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig.savefig(f"{viz_dir}/optimization_progress.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/4] Optimization progress saved")

    # 2. Yield Distribution
    fig = plt.figure(figsize=(8, 4))
    plt.hist(trial_values, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(study.best_value, color='red', linestyle='--', linewidth=2,
                label=f'Best value: {study.best_value:.2f}%')
    plt.xlabel('Yield [%]')
    plt.ylabel('Frequency')
    title = 'Yield Distribution by TPE Exploration'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig.savefig(f"{viz_dir}/yield_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/4] Yield distribution saved")

    # 3. Top 10 Yields
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
    top_values = [t.value for t in top_trials]
    top_indices = list(range(1, 11))

    fig = plt.figure(figsize=(8, 4))
    plt.bar(top_indices, top_values, alpha=0.7, color='green')
    plt.xlabel('Rank')
    plt.ylabel('Yield [%]')
    title = 'Top 10 Highest Yields'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.xticks(top_indices)
    plt.grid(True, alpha=0.3)
    fig.savefig(f"{viz_dir}/top10_yields.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/4] Top 10 yields saved")

    # 4. Yield Statistics
    stats_data = [np.min(trial_values), np.percentile(trial_values, 25),
                  np.median(trial_values), np.percentile(trial_values, 75),
                  np.max(trial_values)]
    stats_labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']

    fig = plt.figure(figsize=(8, 4))
    plt.bar(stats_labels, stats_data, alpha=0.7, color='orange')
    plt.ylabel('Yield [%]')
    title = 'Yield Statistics'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    fig.savefig(f"{viz_dir}/yield_statistics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/4] Yield statistics saved")

    print(f"\nAll visualizations saved to: {viz_dir}")
    return viz_dir


def load_and_visualize(db_path: str, study_name: str, dataset_name: str = ""):
    """
    Load study from database and create visualizations.

    Args:
        db_path: Path to SQLite database file
        study_name: Name of the study to load
        dataset_name: Name of dataset for plot titles

    Returns:
        Path to visualization directory
    """
    try:
        # Load study
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}"
        )

        print(f"Successfully loaded study from database")
        print(f"  Total trials: {len(study.trials)}")
        print(f"  Best yield achieved: {study.best_value:.2f}%")

        # Create visualizations
        output_dir = os.path.dirname(db_path)
        viz_dir = create_visualizations(study, output_dir, dataset_name)

        return viz_dir

    except Exception as e:
        print(f"Error loading study: {e}")
        print("Make sure the optimization has been run first!")
        return None


def print_statistics(study: optuna.Study):
    """
    Print detailed statistics about the optimization.

    Args:
        study: Completed optuna.Study object
    """
    trial_values = [t.value for t in study.trials if t.value is not None]

    print("\n" + "=" * 80)
    print("OPTIMIZATION STATISTICS")
    print("=" * 80)
    print(f"Total trials: {len(trial_values)}")
    print(f"Best yield: {study.best_value:.2f}%")
    print(f"Mean yield: {np.mean(trial_values):.2f}%")
    print(f"Median yield: {np.median(trial_values):.2f}%")
    print(f"Std deviation: {np.std(trial_values):.2f}%")
    print(f"Min yield: {np.min(trial_values):.2f}%")
    print(f"Max yield: {np.max(trial_values):.2f}%")
    print(f"25th percentile: {np.percentile(trial_values, 25):.2f}%")
    print(f"75th percentile: {np.percentile(trial_values, 75):.2f}%")
    print("=" * 80)

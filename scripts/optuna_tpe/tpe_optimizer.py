"""Optuna TPE optimizer for reaction yield optimization."""

import os
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Callable


def create_tpe_sampler(seed: int, n_startup_trials: int = 10, n_ei_candidates: int = 24) -> TPESampler:
    """
    Create TPE sampler with specified parameters.

    Args:
        seed: Random seed for reproducibility
        n_startup_trials: Number of random sampling trials before TPE starts
        n_ei_candidates: Number of candidate samples for expected improvement

    Returns:
        Configured TPESampler instance
    """
    return TPESampler(
        n_startup_trials=n_startup_trials,
        n_ei_candidates=n_ei_candidates,
        seed=seed
    )


def run_tpe_optimization(
    objective_fn: Callable,
    study_name: str,
    db_path: str,
    n_trials: int = 100,
    seed: int = 42,
    n_startup_trials: int = 10,
    n_ei_candidates: int = 24,
    show_progress_bar: bool = True
) -> optuna.Study:
    """
    Run TPE optimization.

    Args:
        objective_fn: Objective function to optimize
        study_name: Name for the optimization study
        db_path: Path to SQLite database file for storing results
        n_trials: Number of optimization trials
        seed: Random seed for reproducibility
        n_startup_trials: Number of random sampling trials before TPE starts
        n_ei_candidates: Number of candidate samples for expected improvement
        show_progress_bar: Whether to display progress bar

    Returns:
        Completed optuna.Study object
    """
    # Create output directory
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Create TPE sampler
    sampler = create_tpe_sampler(seed, n_startup_trials, n_ei_candidates)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=False  # Create new study (fail if exists)
    )

    # Run optimization
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=show_progress_bar)

    return study


def print_optimization_results(study: optuna.Study, db_path: str):
    """
    Print optimization results summary.

    Args:
        study: Completed optuna.Study object
        db_path: Path to database file
    """
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"Best yield: {study.best_value:.2f}%")
    print(f"Total trials: {len(study.trials)}")
    print("\nOptimal reaction conditions:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {db_path}")
    print("=" * 80)

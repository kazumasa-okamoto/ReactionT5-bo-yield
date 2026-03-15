"""Bayesian optimization with GNN + MC Dropout."""

import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from scipy.stats import norm
from torch_geometric.loader import DataLoader

from model_utils import GNNYield, train_epoch, evaluate, mc_dropout_predict


@dataclass
class LoopConfig:
    """Configuration for Bayesian optimization loop."""
    n_initial_random: int = 10  # Initial random samples
    n_bo_iterations: int = 90   # BO iterations
    n_mc_samples: int = 10      # MC Dropout samples

    # Model training settings
    learning_rate: float = 1e-3
    num_epochs: int = 100       # Epochs per training
    weight_decay: float = 1e-4
    batch_size: int = 32
    val_ratio: float = 0.2
    early_stopping_patience: int = 10  # Early stopping patience (0 to disable)

    # Model architecture
    hidden_dim: int = 256
    dropout_rate: float = 0.2
    num_conv_layers: int = 1

    # Output settings
    output_dir: str = "../runs/gnn_mcdropout_100trials_BH"
    log_csv_name: str = "optimization_log.csv"
    save_checkpoints: bool = False
    seed: int = 42


class BayesianOptimizationGNN:
    """Bayesian optimization using GNN + MC Dropout (train from scratch each iteration)."""

    def __init__(self, dataset, node_in_dim, device, config: LoopConfig):
        self.dataset = dataset
        self.node_in_dim = node_in_dim
        self.device = device
        self.config = config

        # Model is created fresh each iteration
        self.model = None

        # Experiment results
        self.experiment_history = []
        self.tried_indices = set()
        self.cumulative_training_data = []  # Cumulative data

        # Prepare output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_csv_path = os.path.join(config.output_dir, config.log_csv_name)
        self._init_csv_log()

        print(f"Dataset size: {len(dataset)}")
        print(f"Initial random samples: {config.n_initial_random}")
        print(f"BO iterations: {config.n_bo_iterations}")

    def _init_csv_log(self):
        """Initialize CSV log file."""
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "trial", "index", "selection_type",
                    "predicted_mean", "predicted_std", "actual_yield",
                    "error_pct", "acquisition_value", "cumulative_data_size",
                    "best_so_far"
                ])

    def _create_fresh_model(self):
        """Create a new model."""
        model = GNNYield(
            node_in_dim=self.node_in_dim,
            hidden_dim=self.config.hidden_dim,
            out_dim=1,
            dropout_rate=self.config.dropout_rate,
            num_conv_layers=self.config.num_conv_layers
        ).to(self.device)
        return model

    def _train_model_from_scratch(self, iteration):
        """Train model from scratch on cumulative data."""
        if len(self.cumulative_training_data) == 0:
            print(f"[Iteration {iteration}] No data, skipping training")
            return None

        print(f"[Iteration {iteration}] Training model from scratch... ({len(self.cumulative_training_data)} samples)")

        # Create new model
        model = self._create_fresh_model()

        # Prepare data
        train_data = self.cumulative_training_data.copy()
        random.Random(self.config.seed + iteration).shuffle(train_data)

        n_total = len(train_data)
        n_val = int(n_total * self.config.val_ratio)

        # Split train/val if we have enough data
        if n_val >= 5:
            val_data = train_data[:n_val]
            train_data = train_data[n_val:]
        else:
            val_data = []

        # Create DataLoaders
        batch_size = min(self.config.batch_size, len(train_data))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None

        # Optimizer and loss
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = self.config.early_stopping_patience

        for epoch in range(self.config.num_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, self.device)

            if val_loader:
                val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, self.device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

                # Early stopping
                if patience > 0 and patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}")

        # Restore best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save checkpoint
        if self.config.save_checkpoints:
            output_dir_iter = os.path.join(self.config.output_dir, f"iteration_{iteration}")
            os.makedirs(output_dir_iter, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir_iter, "model.pt"))

        print(f"[Iteration {iteration}] Training completed (epoch {epoch+1}/{self.config.num_epochs})")
        return model

    def _acquisition_function(self, mean, variance, best_observed_yield, xi=0.01):
        """Expected Improvement acquisition function."""
        if isinstance(variance, (list, np.ndarray)):
            variance = np.array(variance)
            mean = np.array(mean)
            mask = variance <= 0
            if np.any(mask):
                variance[mask] = 1e-8

            std = np.sqrt(variance)
            z = (mean - best_observed_yield - xi) / std
            ei = (mean - best_observed_yield - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei
        else:
            if variance <= 0:
                return 0.0

            std = np.sqrt(variance)
            z = (mean - best_observed_yield - xi) / std
            ei = (mean - best_observed_yield - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

    def _get_best_observed_yield(self):
        """Get best observed yield so far."""
        if len(self.experiment_history) > 0:
            return max([exp['actual_yield'] for exp in self.experiment_history])
        else:
            return 0.0

    def _do_initial_random_selection(self):
        """Initial random sampling."""
        print(f"\n==== Initial Random Sampling ({self.config.n_initial_random} samples) ====")

        # Random selection from all indices
        all_indices = list(range(len(self.dataset)))
        random.Random(self.config.seed).shuffle(all_indices)
        initial_indices = all_indices[:self.config.n_initial_random]

        # Evaluate each sample
        for i, idx in enumerate(initial_indices, 1):
            data = self.dataset[idx]
            actual_yield = data.y.item()  # 0-1 range
            actual_yield_pct = actual_yield * 100  # Convert to percentage (same as bo_yield)

            experiment_result = {
                'iteration': i,
                'index': idx,
                'selection_type': 'random',
                'predicted_mean': None,
                'predicted_std': None,
                'actual_yield': actual_yield_pct,  # Store percentage (same as bo_yield)
                'error_pct': None,
                'acquisition_value': None
            }

            self.experiment_history.append(experiment_result)
            self.tried_indices.add(idx)
            self.cumulative_training_data.append(data)

            print(f"🎲 Random {i}/{self.config.n_initial_random}: Index {idx}, Yield={actual_yield_pct:.2f}%")

        # Log to CSV
        best_so_far = max([exp['actual_yield'] for exp in self.experiment_history])
        for exp in self.experiment_history:
            self._log_experiment_to_csv(exp, best_so_far)

        print(f"\nInitial sampling completed! Best: {best_so_far:.2f}%")
        return best_so_far

    def _select_next_candidate_by_bo(self):
        """Select next candidate using Bayesian optimization."""
        untried_indices = [
            i for i in range(len(self.dataset))
            if i not in self.tried_indices
        ]

        if not untried_indices:
            print("All data points have been tried.")
            return None

        best_yield = self._get_best_observed_yield()  # Percentage (same as bo_yield)
        untried_data = [self.dataset[i] for i in untried_indices]

        print(f"Calculating acquisition function... ({len(untried_data)} candidates)")

        try:
            # mc_dropout_predict returns values in 0-1 range
            pred_means, pred_variances = mc_dropout_predict(
                self.model, untried_data, n_samples=self.config.n_mc_samples, device=self.device
            )

            # Convert to percentage (same as bo_yield: model output * 100)
            pred_means_pct = pred_means * 100
            pred_variances_pct = pred_variances * (100 ** 2)  # variance scales with square

            # Acquisition function uses percentage scale (same as bo_yield)
            acq_values = self._acquisition_function(pred_means_pct, pred_variances_pct, best_yield)
            best_local_idx = np.argmax(acq_values)
            best_global_idx = untried_indices[best_local_idx]

            print(f"🎯 Selected: Index {best_global_idx} (EI: {acq_values[best_local_idx]:.6f})")

            # Return percentage values
            return best_global_idx, pred_means_pct[best_local_idx], pred_variances_pct[best_local_idx]

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return random.choice(untried_indices), None, None

    def _evaluate_candidate(self, idx, pred_mean, pred_variance, iteration, selection_type):
        """Evaluate a candidate (same format as bo_yield)."""
        data = self.dataset[idx]
        actual_yield = data.y.item()  # 0-1 range
        actual_yield_pct = actual_yield * 100  # Convert to percentage

        # Calculate acquisition value and error (all in percentage, same as bo_yield)
        if pred_mean is not None:
            best_yield = self._get_best_observed_yield()  # Percentage
            acquisition_value = self._acquisition_function(pred_mean, pred_variance, best_yield)

            pred_std = np.sqrt(pred_variance) if pred_variance is not None else None
            error_pct = pred_mean - actual_yield_pct
        else:
            acquisition_value = None
            pred_std = None
            error_pct = None

        experiment_result = {
            'iteration': iteration,
            'index': idx,
            'selection_type': selection_type,
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'actual_yield': actual_yield_pct,  # Store percentage (same as bo_yield)
            'error_pct': error_pct,
            'acquisition_value': acquisition_value
        }

        print(f"🔎 Iteration {iteration}: Index {idx}")
        if pred_mean is not None:
            print(f"   📈 Predicted: {pred_mean:.2f}% ± {pred_std:.2f}%")
        print(f"   🧪 Ground truth: {actual_yield_pct:.2f}%")
        if error_pct is not None:
            print(f"   ❗ Error: {error_pct:+.2f}%")
        if acquisition_value is not None:
            print(f"   🎯 Acquisition: {acquisition_value:.6f}")

        return experiment_result

    def _log_experiment_to_csv(self, experiment_result, best_so_far):
        """Log experiment result to CSV (same format as bo_yield)."""
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                experiment_result['iteration'],
                experiment_result['index'],
                experiment_result['selection_type'],
                f"{experiment_result['predicted_mean']:.4f}" if experiment_result['predicted_mean'] is not None else "",
                f"{experiment_result['predicted_std']:.4f}" if experiment_result['predicted_std'] is not None else "",
                f"{experiment_result['actual_yield']:.4f}",  # Percentage (same as bo_yield)
                f"{experiment_result['error_pct']:+.4f}" if experiment_result['error_pct'] is not None else "",
                f"{experiment_result['acquisition_value']:.6f}" if experiment_result['acquisition_value'] is not None else "",
                len(self.cumulative_training_data),
                f"{best_so_far:.4f}"
            ])

    def optimize(self):
        """Run Bayesian optimization (train from scratch each iteration)."""
        print(f"Starting Bayesian optimization (no pre-training, train from scratch each iteration)")
        print("=" * 80)

        # Step 1: Initial random sampling
        self._do_initial_random_selection()

        # Step 2: Train initial model
        print(f"\n==== Initial Model Training ====")
        self.model = self._train_model_from_scratch(iteration=0)

        # Step 3: Bayesian optimization loop
        print(f"\n==== Starting Bayesian Optimization Loop ====")
        for bo_iter in range(1, self.config.n_bo_iterations + 1):
            iteration = self.config.n_initial_random + bo_iter
            print(f"\n---- Iteration {iteration} (BO {bo_iter}/{self.config.n_bo_iterations}) ----")

            # Select next candidate
            result = self._select_next_candidate_by_bo()
            if result is None:
                print("All data points tried.")
                break

            idx, pred_mean, pred_variance = result

            # Evaluate candidate
            exp_result = self._evaluate_candidate(idx, pred_mean, pred_variance, iteration, 'bayesian')

            self.experiment_history.append(exp_result)
            self.tried_indices.add(idx)
            self.cumulative_training_data.append(self.dataset[idx])

            # Progress
            current_best = max([exp['actual_yield'] for exp in self.experiment_history])
            print(f"   💡 Current best: {current_best:.2f}%")

            # Log to CSV
            self._log_experiment_to_csv(exp_result, current_best)

            # Retrain model with cumulative data
            print(f"\nRetraining (cumulative data: {len(self.cumulative_training_data)} samples)")
            self.model = self._train_model_from_scratch(iteration=iteration)

            print("-" * 80)

        print(f"\nOptimization completed! Total trials: {len(self.experiment_history)}")

        # Final results
        if self.experiment_history:
            best_exp = max(self.experiment_history, key=lambda x: x['actual_yield'])
            print(f"🏆 Best yield: {best_exp['actual_yield']:.2f}%")
            print(f"🏆 Best index: {best_exp['index']}")
            print(f"🏆 Found at iteration: {best_exp['iteration']}")
            print(f"🏆 Selection method: {best_exp['selection_type']}")
            return best_exp
        else:
            return None

    def get_optimization_summary(self):
        """Get optimization summary statistics (same format as bo_yield)."""
        if not self.experiment_history:
            return {}

        actual_yields = [exp['actual_yield'] for exp in self.experiment_history]
        errors = [exp['error_pct'] for exp in self.experiment_history if exp['error_pct'] is not None]

        # Separate random and Bayesian statistics
        random_exps = [exp for exp in self.experiment_history if exp['selection_type'] == 'random']
        bayesian_exps = [exp for exp in self.experiment_history if exp['selection_type'] == 'bayesian']

        return {
            'total_trials': len(self.experiment_history),
            'random_trials': len(random_exps),
            'bayesian_trials': len(bayesian_exps),
            'max_yield': max(actual_yields),
            'mean_yield': np.mean(actual_yields),
            'std_yield': np.std(actual_yields),
            'mae_error': np.mean(np.abs(errors)) if errors else None,
            'rmse_error': np.sqrt(np.mean(np.array(errors) ** 2)) if errors else None,
            'coverage': len(self.tried_indices) / len(self.dataset) * 100,
            'training_data_size': len(self.cumulative_training_data),
            'log_path': self.log_csv_path
        }

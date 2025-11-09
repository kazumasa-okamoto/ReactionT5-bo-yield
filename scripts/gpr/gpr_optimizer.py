"""Gaussian Process Regression Bayesian Optimization for reaction yield."""

import os
import csv
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt


@dataclass
class GPRConfig:
    """Configuration for GPR Bayesian optimization."""
    n_trials: int = 100
    radius: int = 2
    n_bits: int = 2048
    output_dir: str = "runs/gpr_yield"
    log_csv_name: str = "optimization_log.csv"
    study_seed: int = 42


class GaussianProcessBayesianOptimization:
    """
    Gaussian Process Regression Bayesian Optimization
    Uses Morgan Fingerprint as features
    """
    
    def __init__(self, reactant_list, reagent_list, product_dict, true_yield_dict, 
                 fingerprint_dict, config: GPRConfig):
        self.reactant_list = reactant_list
        self.reagent_list = reagent_list
        self.product_dict = product_dict
        self.true_yield_dict = true_yield_dict
        self.fingerprint_dict = fingerprint_dict
        self.config = config
        
        # Pre-compute all valid combinations
        self.valid_combinations = []
        for reactant in reactant_list:
            for reagent in reagent_list:
                if (reactant, reagent) in product_dict:
                    self.valid_combinations.append((reactant, reagent))
        
        print(f"Valid combinations: {len(self.valid_combinations)}")
        
        # Initialize Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6,
            random_state=config.study_seed
        )
        
        # Experiment tracking
        self.experiment_history = []
        self.tried_combinations = set()
        self.X_train = []
        self.y_train = []
        
        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_csv_path = os.path.join(config.output_dir, config.log_csv_name)
        self._init_csv_log()
    
    def _init_csv_log(self):
        """Initialize CSV log file."""
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "trial", "reactant", "reagent", "product",
                    "predicted_mean", "predicted_std", "actual_yield",
                    "error_pct", "acquisition_value"
                ])
    
    def _predict_yield_with_uncertainty(self, fingerprints):
        """Predict yield and uncertainty using GPR."""
        if len(self.X_train) == 0:
            # Initial random predictions
            n_samples = fingerprints.shape[0] if len(fingerprints.shape) > 1 else 1
            return np.ones(n_samples) * 50.0, np.ones(n_samples) * 30.0
        
        mean, std = self.gp_model.predict(fingerprints, return_std=True)
        return mean, std
    
    def _acquisition_function(self, mean, std, best_observed_yield, xi=0.01):
        """Expected Improvement acquisition function."""
        if isinstance(std, np.ndarray):
            std = np.maximum(std, 1e-9)
        else:
            std = max(std, 1e-9)
        
        z = (mean - best_observed_yield - xi) / std
        ei = (mean - best_observed_yield - xi) * norm.cdf(z) + std * norm.pdf(z)
        return ei
    
    def _get_best_observed_yield(self):
        """Get best observed yield so far."""
        if len(self.experiment_history) > 0:
            return max([exp['actual_yield'] for exp in self.experiment_history])
        else:
            return 0.0
    
    def _select_next_candidate(self):
        """Select next candidate based on acquisition function."""
        untried_combinations = [
            combo for combo in self.valid_combinations 
            if combo not in self.tried_combinations
        ]
        
        if not untried_combinations:
            print("All combinations tried.")
            return None
        
        best_yield = self._get_best_observed_yield()
        
        # Create fingerprints list
        fingerprints = []
        for reactant, reagent in untried_combinations:
            product = self.product_dict[(reactant, reagent)]
            fp = self.fingerprint_dict[(reactant, reagent, product)]
            fingerprints.append(fp)
        
        fingerprints = np.array(fingerprints)
        
        print(f"Calculating acquisition values for {len(untried_combinations)} candidates...")
        
        try:
            pred_means, pred_stds = self._predict_yield_with_uncertainty(fingerprints)
            acq_values = self._acquisition_function(pred_means, pred_stds, best_yield)
            
            best_idx = np.argmax(acq_values)
            selected_combo = untried_combinations[best_idx]
            
            print(f"[Selection] Candidate {selected_combo} (EI={acq_values[best_idx]:.4f})")
            
            return selected_combo
            
        except Exception as e:
            print(f"[Error] Acquisition calculation failed: {e}")
            import random
            return random.choice(untried_combinations)
    
    def _evaluate_candidate(self, reactant, reagent, trial_num):
        """Evaluate candidate (run experiment)."""
        product = self.product_dict[(reactant, reagent)]
        fp = self.fingerprint_dict[(reactant, reagent, product)]
        
        try:
            pred_mean, pred_std = self._predict_yield_with_uncertainty(fp.reshape(1, -1))
            pred_mean = float(pred_mean[0])
            pred_std = float(pred_std[0])
        except Exception as e:
            print(f"[Error] Prediction failed: {e}")
            return None
        
        key = (reactant, reagent, product)
        actual_yield = self.true_yield_dict.get(key, 0.0)
        
        best_yield = self._get_best_observed_yield()
        acquisition_value = float(self._acquisition_function(pred_mean, pred_std, best_yield))
        
        error_pct = pred_mean - actual_yield
        
        experiment_result = {
            'trial': trial_num,
            'reactant': reactant,
            'reagent': reagent,
            'product': product,
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'actual_yield': actual_yield,
            'error_pct': error_pct,
            'acquisition_value': acquisition_value
        }
        
        print(f"[Trial {trial_num}] {reactant} + {reagent} -> {product}")
        print(f"   Predicted yield: {pred_mean:.2f}% +/- {pred_std:.2f}%")
        print(f"   Ground truth: {actual_yield:.2f}%")
        print(f"   Prediction error: {error_pct:+.2f}%")
        print(f"   Acquisition score: {acquisition_value:.4f}")
        
        return experiment_result
    
    def _update_model(self, fingerprint, yield_value):
        """Update model with new data."""
        self.X_train.append(fingerprint)
        self.y_train.append(yield_value)
        
        X_train_array = np.array(self.X_train)
        y_train_array = np.array(self.y_train)
        
        print(f"   Updating model with {len(self.X_train)} data points")
        self.gp_model.fit(X_train_array, y_train_array)
    
    def _log_experiment_to_csv(self, experiment_result):
        """Log experiment result to CSV."""
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                experiment_result['trial'],
                experiment_result['reactant'],
                experiment_result['reagent'],
                experiment_result['product'],
                f"{experiment_result['predicted_mean']:.6f}",
                f"{experiment_result['predicted_std']:.6f}",
                f"{experiment_result['actual_yield']:.6f}",
                f"{experiment_result['error_pct']:+.6f}",
                f"{experiment_result['acquisition_value']:.6f}"
            ])
    
    def optimize(self):
        """Execute Bayesian optimization."""
        print(f"Starting GPR Bayesian optimization ({self.config.n_trials} trials)")
        print("=" * 60)
        
        for trial in range(1, self.config.n_trials + 1):
            candidate = self._select_next_candidate()
            if candidate is None:
                print("All combinations tried.")
                break
            
            reactant, reagent = candidate
            product = self.product_dict[(reactant, reagent)]
            
            result = self._evaluate_candidate(reactant, reagent, trial)
            if result is None:
                continue
            
            self.experiment_history.append(result)
            self.tried_combinations.add((reactant, reagent))
            
            self._log_experiment_to_csv(result)
            
            fp = self.fingerprint_dict[(reactant, reagent, product)]
            self._update_model(fp, result['actual_yield'])
            
            current_best = max([exp['actual_yield'] for exp in self.experiment_history])
            print(f"   Current best yield: {current_best:.2f}%")
            print("-" * 60)
        
        print(f"\nOptimization complete! Total trials: {len(self.experiment_history)}")
        
        if self.experiment_history:
            best_exp = max(self.experiment_history, key=lambda x: x['actual_yield'])
            print(f"Best yield achieved: {best_exp['actual_yield']:.2f}%")
            print(f"Best combination: {best_exp['reactant']} + {best_exp['reagent']} -> {best_exp['product']}")
            return best_exp
        else:
            return None
    
    def get_optimization_summary(self):
        """Get optimization summary statistics."""
        if not self.experiment_history:
            return {}

        actual_yields = [exp['actual_yield'] for exp in self.experiment_history]
        errors = [exp['error_pct'] for exp in self.experiment_history]

        return {
            'total_trials': len(self.experiment_history),
            'max_yield': max(actual_yields),
            'mean_yield': np.mean(actual_yields),
            'std_yield': np.std(actual_yields),
            'mae_error': np.mean(np.abs(errors)),
            'rmse_error': np.sqrt(np.mean(np.array(errors)**2)),
            'coverage': len(self.tried_combinations) / len(self.valid_combinations) * 100,
            'log_path': self.log_csv_path
        }

    def save_visualization(self):
        """Visualize optimization results and save to files."""
        if not self.experiment_history:
            print("No experiment history available for visualization.")
            return

        save_dir = os.path.join(self.config.output_dir, "visualization")
        os.makedirs(save_dir, exist_ok=True)

        trials = [exp['trial'] for exp in self.experiment_history]
        actual_yields = [exp['actual_yield'] for exp in self.experiment_history]
        predicted_means = [exp['predicted_mean'] for exp in self.experiment_history]
        predicted_stds = [exp['predicted_std'] for exp in self.experiment_history]
        errors = [exp['error_pct'] for exp in self.experiment_history]

        # Calculate cumulative maximum
        cumulative_max = np.maximum.accumulate(actual_yields)

        # 1. Optimization Progress
        fig = plt.figure(figsize=(8, 5))
        plt.plot(trials, actual_yields, 'o-', alpha=0.6, label='Actual Yield', markersize=6)
        plt.plot(trials, cumulative_max, 'r-', linewidth=3, label='Best So Far')
        plt.xlabel('Trial')
        plt.ylabel('Yield [%]')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'optimization_progress.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/optimization_progress.png")

        # 2. Parity Plot
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(actual_yields, predicted_means, s=50, alpha=0.6)
        lims = [0, 100]
        plt.plot(lims, lims, 'r--', label='Perfect Prediction')
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel('Actual Yield [%]')
        plt.ylabel('Predicted Yield [%]')
        plt.title('Parity Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parity.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/parity.png")

        # 3. Prediction Uncertainty
        fig = plt.figure(figsize=(8, 5))
        plt.errorbar(trials, predicted_means, yerr=predicted_stds,
                    fmt='o', alpha=0.6, capsize=5, label='Prediction ± Uncertainty')
        plt.plot(trials, actual_yields, 'ro', alpha=0.6, label='Actual Yield')
        plt.xlabel('Trial')
        plt.ylabel('Yield [%]')
        plt.title('Prediction Uncertainty')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'uncertainty.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/uncertainty.png")

        # 4. Error Trend
        fig = plt.figure(figsize=(8, 5))
        plt.plot(trials, errors, 'o-', alpha=0.6)
        plt.axhline(0, color='r', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Trial')
        plt.ylabel('Prediction Error [%]')
        plt.title('Prediction Error Trend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_trend.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/error_trend.png")

        # 5. Error Histogram
        fig = plt.figure(figsize=(7, 5))
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='r', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Prediction Error [%]')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_hist.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/error_hist.png")

        # 6. Uncertainty vs Actual Yield
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(predicted_stds, actual_yields, alpha=0.6)
        plt.xlabel('Prediction Uncertainty (Std)')
        plt.ylabel('Actual Yield [%]')
        plt.title('Uncertainty vs Actual Yield')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'uncertainty_vs_yield.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_dir}/uncertainty_vs_yield.png")

        print(f"\nVisualization complete! All plots saved to: {save_dir}")

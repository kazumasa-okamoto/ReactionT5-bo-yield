"""Bayesian optimization with fine-tuning for reaction yield optimization."""

import os
import csv
import time
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import norm
from transformers import TrainingArguments

from model_utils import mc_dropout_inference
from training_utils import CollatorForYield, YieldTrainer, YieldDataset


@dataclass
class LoopConfig:
    """Configuration for Bayesian optimization + fine-tuning loop."""
    # Bayesian optimization settings
    n_rounds: int = 10
    trials_per_round: int = 10
    n_mc_samples: int = 50
    batch_size_prediction: int = 32

    # Fine-tuning settings
    learning_rate: float = 5e-4
    epochs_per_round: int = 3
    weight_decay: float = 0.01
    max_length: int = 512
    batch_size_train: int = 16
    batch_size_eval: int = 32
    val_ratio: float = 0.2

    # Output settings
    output_dir: str = "runs/bayes_finetune"
    log_csv_name: str = "optimization_log.csv"
    save_checkpoints: bool = True

    # Other
    study_seed: int = 42


class BayesianOptimizationWithFineTuning:
    """Bayesian optimization with MC Dropout + fine-tuning loop."""

    def __init__(
        self,
        model,
        tokenizer,
        reactant_list: List[str],
        reagent_list: List[str],
        product_dict: Dict[Tuple[str, str], str],
        true_yield_dict: Dict[Tuple[str, str, str], float],
        config: LoopConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reactant_list = reactant_list
        self.reagent_list = reagent_list
        self.product_dict = product_dict
        self.true_yield_dict = true_yield_dict
        self.config = config

        # Pre-compute all valid combinations
        self.valid_combinations = []
        for reactant in reactant_list:
            for reagent in reagent_list:
                if (reactant, reagent) in product_dict:
                    self.valid_combinations.append((reactant, reagent))

        print(f"Valid combinations: {len(self.valid_combinations)}")

        # Record experiment results
        self.experiment_history = []
        self.tried_combinations = set()
        self.cumulative_training_data = []  # Cumulative data for fine-tuning

        # Prepare output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_csv_path = os.path.join(config.output_dir, config.log_csv_name)
        self._init_csv_log()

    def _init_csv_log(self):
        """Initialize CSV log file."""
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "round", "trial", "reactant", "reagent", "product",
                    "reaction_smiles", "predicted_mean", "predicted_std", "actual_yield",
                    "error_pct", "acquisition_value", "was_used_for_ft",
                    "round_best_pred", "round_best_true"
                ])

    def _predict_yield_with_uncertainty(self, reaction_smiles_list):
        """Batch MC Dropout yield prediction."""
        result = mc_dropout_inference(
            self.model, self.tokenizer, reaction_smiles_list,
            n_samples=self.config.n_mc_samples,
            batch_size=self.config.batch_size_prediction
        )
        return result['mean'], result['variance']

    def _acquisition_function(self, mean, variance, best_observed_yield, xi=0.01):
        """Expected Improvement (EI) acquisition function."""
        if isinstance(variance, (list, np.ndarray)):
            # Batch processing
            variance = np.array(variance)
            mean = np.array(mean)
            mask = variance <= 0
            if np.any(mask):
                variance[mask] = 1e-8  # Replace with small value

            std = np.sqrt(variance)
            z = (mean - best_observed_yield - xi) / std
            ei = (mean - best_observed_yield - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei
        else:
            # Single value processing
            if variance <= 0:
                return 0.0

            std = np.sqrt(variance)
            z = (mean - best_observed_yield - xi) / std
            ei = (mean - best_observed_yield - xi) * norm.cdf(z) + std * norm.pdf(z)
            return ei

    def _get_best_observed_yield(self):
        """Get the best yield observed so far."""
        if len(self.experiment_history) > 0:
            return max([exp['actual_yield'] for exp in self.experiment_history])
        else:
            return 0.0

    def _select_next_candidate(self):
        """Select next candidate based on acquisition function (batch processing)."""
        # Get untried combinations
        untried_combinations = [
            combo for combo in self.valid_combinations
            if combo not in self.tried_combinations
        ]

        if not untried_combinations:
            print("All combinations have been tried.")
            return None

        # Calculate acquisition function in batch
        best_yield = self._get_best_observed_yield()

        # Create list of reaction SMILES
        reaction_smiles_list = []
        for reactant, reagent in untried_combinations:
            product = self.product_dict[(reactant, reagent)]
            reaction_smiles = f"REACTANT:{reactant}REAGENT:{reagent}PRODUCT:{product}"
            reaction_smiles_list.append(reaction_smiles)

        print(f"Calculating acquisition function values... ({len(reaction_smiles_list)} candidates)")

        try:
            # Batch prediction
            pred_means, pred_variances = self._predict_yield_with_uncertainty(reaction_smiles_list)

            # Calculate acquisition function values
            acq_values = self._acquisition_function(pred_means, pred_variances, best_yield)

            # Combine results
            acquisition_scores = []
            for i, (combo, mean, var, acq) in enumerate(zip(untried_combinations, pred_means, pred_variances, acq_values)):
                acquisition_scores.append({
                    'combination': combo,
                    'acquisition_value': float(acq),
                    'predicted_mean': float(mean),
                    'predicted_variance': float(var)
                })

        except Exception as e:
            print(f"[Error] Batch prediction failed: {e}")
            # Random selection on error
            combo = np.random.choice(len(untried_combinations))
            return untried_combinations[combo]

        if not acquisition_scores:
            # Random selection if acquisition function cannot be calculated
            combo = np.random.choice(len(untried_combinations))
            return untried_combinations[combo]

        # Sort by acquisition function value
        acquisition_scores.sort(key=lambda x: x['acquisition_value'], reverse=True)

        # Select highest acquisition value
        selected = acquisition_scores[0]
        print(f"[Selection] Candidate {selected['combination']} (EI={selected['acquisition_value']:.4f})")

        return selected['combination']

    def _evaluate_candidate(self, reactant, reagent, round_num, trial_num):
        """Evaluate candidate (run experiment)."""
        product = self.product_dict[(reactant, reagent)]
        reaction_smiles = f"REACTANT:{reactant}REAGENT:{reagent}PRODUCT:{product}"

        # MC Dropout prediction
        try:
            pred_mean, pred_variance = self._predict_yield_with_uncertainty(reaction_smiles)
            pred_std = np.sqrt(pred_variance)
        except Exception as e:
            print(f"[Error] Prediction failed: {e}")
            return None

        # Get actual yield
        key = (reactant, reagent, product)
        actual_yield = self.true_yield_dict.get(key)

        if actual_yield is None:
            print(f"[Warning] No ground truth for: {reactant} + {reagent} -> {product}")
            actual_yield_pct = 0.0
        else:
            actual_yield_pct = actual_yield * 100  # Convert to percentage

        # Calculate acquisition value
        best_yield = self._get_best_observed_yield()
        acquisition_value = self._acquisition_function(pred_mean, pred_variance, best_yield)

        # Calculate error
        error_pct = pred_mean - actual_yield_pct if actual_yield is not None else None

        # Record experiment result
        experiment_result = {
            'round': round_num,
            'trial': trial_num,
            'reactant': reactant,
            'reagent': reagent,
            'product': product,
            'reaction_smiles': reaction_smiles,
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'actual_yield': actual_yield_pct,
            'error_pct': error_pct,
            'acquisition_value': acquisition_value
        }

        # Output
        print(f"[Round {round_num} Trial {trial_num}] {reactant} + {reagent} -> {product}")
        print(f"   Predicted: {pred_mean:.2f}% +/- {pred_std:.2f}%")
        print(f"   Ground truth: {actual_yield_pct:.2f}%" if actual_yield is not None else "   Ground truth: None")
        if error_pct is not None:
            print(f"   Error: {error_pct:+.2f}%")
        print(f"   Acquisition score: {acquisition_value:.4f}")

        return experiment_result

    def _fine_tune_model(self, round_num):
        """Fine-tune the model."""
        if len(self.cumulative_training_data) == 0:
            print(f"[Round {round_num}] No labeled data, skipping fine-tuning.")
            return

        print(f"[Round {round_num}] Starting fine-tuning... ({len(self.cumulative_training_data)} data points)")

        # Prepare data
        texts = [data["reaction_smiles"] for data in self.cumulative_training_data]
        labels = [data["actual_yield"] for data in self.cumulative_training_data]

        # Train/eval split
        indices = list(range(len(texts)))
        random.Random(self.config.study_seed + round_num).shuffle(indices)

        n_total = len(indices)
        n_val = int(n_total * self.config.val_ratio)

        if n_val >= 5:
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
        else:
            val_indices = []
            train_indices = indices

        # Create datasets
        def subset(lst, sel_indices):
            return [lst[i] for i in sel_indices]

        train_dataset = YieldDataset(
            subset(texts, train_indices),
            subset(labels, train_indices),
            self.tokenizer,
            max_length=self.config.max_length
        )

        eval_dataset = None
        if len(val_indices) > 0:
            eval_dataset = YieldDataset(
                subset(texts, val_indices),
                subset(labels, val_indices),
                self.tokenizer,
                max_length=self.config.max_length
            )

        # Trainer configuration
        output_dir_round = os.path.join(self.config.output_dir, f"round_{round_num}")
        training_args = TrainingArguments(
            output_dir=output_dir_round,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs_per_round,
            per_device_train_batch_size=min(self.config.batch_size_train, max(1, len(train_dataset))),
            per_device_eval_batch_size=self.config.batch_size_eval,
            weight_decay=self.config.weight_decay,
            logging_strategy="epoch",
            logging_first_step=True,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="no",
            report_to="tensorboard",
            logging_dir=os.path.join(output_dir_round, "logs"),
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False
        )

        trainer = YieldTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=CollatorForYield(self.tokenizer),
        )

        train_output = trainer.train()

        if self.config.save_checkpoints:
            trainer.save_model(output_dir_round)

        print(f"[Round {round_num}] Fine-tuning completed")

    def _log_experiment_to_csv(self, experiment_result, round_best_pred, round_best_true):
        """Log experiment result to CSV."""
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time()),
                experiment_result['round'],
                experiment_result['trial'],
                experiment_result['reactant'],
                experiment_result['reagent'],
                experiment_result['product'],
                experiment_result['reaction_smiles'],
                f"{experiment_result['predicted_mean']:.6f}",
                f"{experiment_result['predicted_std']:.6f}",
                f"{experiment_result['actual_yield']:.6f}" if experiment_result['actual_yield'] is not None else "",
                f"{experiment_result['error_pct']:+.6f}" if experiment_result['error_pct'] is not None else "",
                f"{experiment_result['acquisition_value']:.6f}",
                "1" if experiment_result['actual_yield'] is not None else "0",
                f"{round_best_pred:.6f}" if round_best_pred is not None else "",
                f"{round_best_true:.6f}" if round_best_true is not None else ""
            ])

    def optimize_with_finetuning(self):
        """Execute Bayesian optimization + fine-tuning loop."""
        print(f"Starting Bayesian optimization + fine-tuning loop ({self.config.n_rounds} rounds, {self.config.trials_per_round} trials per round)")
        print("=" * 80)

        for round_num in range(1, self.config.n_rounds + 1):
            print(f"\n==== Round {round_num}/{self.config.n_rounds} ====")

            round_experiments = []

            # Bayesian optimization within round
            for trial_num in range(1, self.config.trials_per_round + 1):
                # Select next candidate
                candidate = self._select_next_candidate()
                if candidate is None:
                    print("All combinations tried.")
                    break

                reactant, reagent = candidate

                # Evaluate candidate
                result = self._evaluate_candidate(reactant, reagent, round_num, trial_num)
                if result is None:
                    continue

                # Record result
                self.experiment_history.append(result)
                round_experiments.append(result)
                self.tried_combinations.add((reactant, reagent))

                # Add labeled data to cumulative data
                if result['actual_yield'] is not None:
                    self.cumulative_training_data.append({
                        'reaction_smiles': result['reaction_smiles'],
                        'actual_yield': result['actual_yield']
                    })

                # Progress display
                current_best = max([exp['actual_yield'] for exp in self.experiment_history])
                print(f"   Current best yield: {current_best:.2f}%")
                print("-" * 60)

            # Round statistics
            if round_experiments:
                round_best_pred = max([exp['predicted_mean'] for exp in round_experiments])
                round_best_true = max([exp['actual_yield'] for exp in round_experiments if exp['actual_yield'] is not None], default=None)

                # Log to CSV
                for exp in round_experiments:
                    self._log_experiment_to_csv(exp, round_best_pred, round_best_true)

                print(f"\n[Round {round_num}] Statistics:")
                print(f"  Trials: {len(round_experiments)}")
                print(f"  Best predicted yield: {round_best_pred:.2f}%")
                if round_best_true is not None:
                    print(f"  Best actual yield: {round_best_true:.2f}%")

            # Fine-tune model (except final round)
            if round_num < self.config.n_rounds:
                self._fine_tune_model(round_num)

            print(f"[Round {round_num}] Completed")

        # Final results
        print(f"\nOptimization complete! Total trials: {len(self.experiment_history)}")

        if self.experiment_history:
            best_exp = max(self.experiment_history, key=lambda x: x['actual_yield'])
            print(f"Best yield achieved: {best_exp['actual_yield']:.2f}%")
            print(f"Best combination: {best_exp['reactant']} + {best_exp['reagent']} -> {best_exp['product']}")
            print(f"Discovered in round: {best_exp['round']}")
            return best_exp
        else:
            return None

    def get_optimization_summary(self):
        """Get optimization summary statistics."""
        if not self.experiment_history:
            return {}

        actual_yields = [exp['actual_yield'] for exp in self.experiment_history]
        predicted_means = [exp['predicted_mean'] for exp in self.experiment_history]
        errors = [exp['error_pct'] for exp in self.experiment_history if exp['error_pct'] is not None]

        # Round-wise statistics
        round_stats = {}
        for round_num in range(1, self.config.n_rounds + 1):
            round_exps = [exp for exp in self.experiment_history if exp['round'] == round_num]
            if round_exps:
                round_yields = [exp['actual_yield'] for exp in round_exps]
                round_stats[round_num] = {
                    'trials': len(round_exps),
                    'max_yield': max(round_yields),
                    'mean_yield': np.mean(round_yields)
                }

        return {
            'total_trials': len(self.experiment_history),
            'max_yield': max(actual_yields),
            'mean_yield': np.mean(actual_yields),
            'std_yield': np.std(actual_yields),
            'mae_error': np.mean(np.abs(errors)) if errors else None,
            'rmse_error': np.sqrt(np.mean(np.array(errors) ** 2)) if errors else None,
            'coverage': len(self.tried_combinations) / len(self.valid_combinations) * 100,
            'training_data_size': len(self.cumulative_training_data),
            'round_stats': round_stats,
            'log_path': self.log_csv_path
        }

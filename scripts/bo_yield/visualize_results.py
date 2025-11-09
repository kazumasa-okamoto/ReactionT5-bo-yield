"""Visualization utilities for optimization results."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def visualize_logs(csv_path: str, out_dir: str = None, show: bool = False, dpi: int = 180):
    """
    Visualize optimization logs from CSV.

    Args:
        csv_path: Path to CSV file
        out_dir: Output directory (if None, uses dirname of csv_path)
        show: Whether to show plots
        dpi: Image resolution

    Returns:
        Dictionary with visualization results
    """
    # Output directory
    root = out_dir or (os.path.dirname(csv_path) or ".")
    save_dir = os.path.join(root, "visualization")
    os.makedirs(save_dir, exist_ok=True)

    # Load & format
    df = pd.read_csv(csv_path)
    to_num_cols = [
        "round", "trial", "predicted_mean", "predicted_std", "actual_yield",
        "error_pct", "acquisition_value", "round_best_pred", "round_best_true"
    ]
    for c in to_num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Data with ground truth
    df_obs = df.dropna(subset=["actual_yield"]) if "actual_yield" in df.columns else pd.DataFrame()
    # Recalculate error_pct if missing/NaN
    if not df_obs.empty and ("error_pct" not in df_obs.columns or df_obs["error_pct"].isna().any()):
        df_obs["error_pct"] = df_obs["predicted_mean"] - df_obs["actual_yield"]

    # ===== Overall metrics =====
    overall = {}
    if not df_obs.empty:
        y = df_obs["actual_yield"].to_numpy()
        yhat = df_obs["predicted_mean"].to_numpy()
        err = yhat - y
        overall = {
            "n": int(len(df_obs)),
            "mae_pct": float(np.mean(np.abs(err))),
            "rmse_pct": float(np.sqrt(np.mean(err ** 2))),
            "bias_pct": float(np.mean(err)),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if len(df_obs) > 1 else np.nan,
        }
    pd.DataFrame([overall]).to_csv(os.path.join(save_dir, "metrics_overall.csv"), index=False)

    # ===== Round-wise metrics (MAE/RMSE) =====
    metrics_by_round = []
    if not df_obs.empty and "round" in df_obs.columns:
        for r, d in df_obs.groupby("round", dropna=True):
            y = d["actual_yield"].to_numpy()
            yhat = d["predicted_mean"].to_numpy()
            err = yhat - y
            metrics_by_round.append({
                "round": int(r),
                "n": int(len(d)),
                "mae_pct": float(np.mean(np.abs(err))),
                "rmse_pct": float(np.sqrt(np.mean(err ** 2))),
                "bias_pct": float(np.mean(err)),
                "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if len(d) > 1 else np.nan,
            })
    mdf = (pd.DataFrame(metrics_by_round)
           .sort_values("round")
           if metrics_by_round else pd.DataFrame(columns=["round", "n", "mae_pct", "rmse_pct"]))
    mdf.to_csv(os.path.join(save_dir, "metrics_by_round.csv"), index=False)

    # ===== Plot: Parity =====
    if not df_obs.empty:
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(df_obs["actual_yield"], df_obs["predicted_mean"], s=18, alpha=0.65)
        lims = [0, 100]
        plt.plot(lims, lims, linestyle="--", color="red", label="Perfect Prediction")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.xlabel("True Yield [%]")
        plt.ylabel("Predicted Yield [%]")
        plt.title("Parity: Prediction vs Truth")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "parity.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Error histogram =====
    if not df_obs.empty and not df_obs["error_pct"].dropna().empty:
        fig = plt.figure(figsize=(6, 4))
        plt.hist(df_obs["error_pct"].dropna().to_numpy(), bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel("Prediction Error [%]  (pred - true)")
        plt.ylabel("Count")
        plt.title("Error Histogram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "error_hist.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Best true yield per round =====
    if not df_obs.empty and "round" in df_obs.columns:
        best_by_round = df_obs.groupby("round")["actual_yield"].max()
        fig = plt.figure(figsize=(6, 4))
        plt.plot(best_by_round.index, best_by_round.values, marker="o", linewidth=2)
        plt.xlabel("Round")
        plt.ylabel("Best Observed True Yield [%]")
        plt.title("Best True Yield per Round")
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "best_true_per_round.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Uncertainty vs actual yield =====
    if not df_obs.empty and "predicted_std" in df_obs.columns:
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(df_obs["predicted_std"], df_obs["actual_yield"], alpha=0.6)
        plt.xlabel("Prediction Uncertainty (Std)")
        plt.ylabel("Actual Yield [%]")
        plt.title("Uncertainty vs Actual Yield")
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "uncertainty_vs_yield.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Calibration =====
    if not df_obs.empty:
        bins = np.linspace(0, 100, 11)  # 10 bins
        cut = pd.cut(df_obs["predicted_mean"], bins, include_lowest=True)
        calib = df_obs.groupby(cut, observed=False).agg(
            pred_mean=("predicted_mean", "mean"),
            true_mean=("actual_yield", "mean"),
            n=("actual_yield", "size")
        ).dropna()
        if not calib.empty:
            fig = plt.figure(figsize=(6, 4))
            lims = [0, 100]
            plt.plot(calib["pred_mean"], calib["true_mean"], marker="o", linewidth=2)
            plt.plot(lims, lims, linestyle="--", color="red")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.xlabel("Predicted Mean (per bin) [%]")
            plt.ylabel("Observed Mean (per bin) [%]")
            plt.title("Calibration Curve")
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "calibration.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            calib.to_csv(os.path.join(save_dir, "calibration_table.csv"))

    # ===== Plot: Round-wise boxplot (true/pred) =====
    if not df_obs.empty and "round" in df_obs.columns:
        rounds = sorted(df_obs["round"].dropna().unique())
        if len(rounds) > 0:
            fig = plt.figure(figsize=(7, 4))
            pos = np.array(rounds, dtype=float)
            data_true = [df_obs[df_obs["round"] == r]["actual_yield"].to_numpy() for r in rounds]
            data_pred = [df_obs[df_obs["round"] == r]["predicted_mean"].to_numpy() for r in rounds]
            bp1 = plt.boxplot(data_true, positions=pos - 0.15, widths=0.25, patch_artist=True)
            bp2 = plt.boxplot(data_pred, positions=pos + 0.15, widths=0.25, patch_artist=True)

            # Color
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            for patch in bp2['boxes']:
                patch.set_facecolor('lightcoral')

            plt.xticks(rounds)
            plt.xlabel("Round")
            plt.ylabel("Yield [%]")
            plt.title("Distributions per Round (True vs Pred)")
            plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['True', 'Predicted'], loc='upper right')
            fig.savefig(os.path.join(save_dir, "round_box.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Plot: Round-wise error boxplot =====
    if not df_obs.empty and "round" in df_obs.columns:
        rounds = sorted(df_obs["round"].dropna().unique())
        if len(rounds) > 0:
            fig = plt.figure(figsize=(7, 4))
            data_err = [df_obs[df_obs["round"] == r]["error_pct"].dropna().to_numpy() for r in rounds]
            plt.boxplot(data_err, positions=np.array(rounds, dtype=float), widths=0.5, patch_artist=True)
            plt.axhline(0.0, linestyle="--", color="red", label="Perfect Prediction")
            plt.xticks(rounds)
            plt.xlabel("Round")
            plt.ylabel("Prediction Error (pred - true) [%]")
            plt.title("Prediction Error by Round (Boxplot)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "error_box_by_round.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Plot: MAE/RMSE by round =====
    if not mdf.empty:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(mdf["round"], mdf["mae_pct"], marker="o", label="MAE [%]", linewidth=2)
        plt.plot(mdf["round"], mdf["rmse_pct"], marker="o", label="RMSE [%]", linewidth=2)
        plt.xlabel("Round")
        plt.ylabel("Error [%]")
        plt.title("Prediction Error by Round")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.savefig(os.path.join(save_dir, "error_by_round.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Round best pred/true =====
    if "round_best_pred" in df.columns:
        best_df = df.dropna(subset=["round", "round_best_pred"]).groupby("round").agg(
            best_pred=("round_best_pred", "max"),
            best_true=("round_best_true", "max")
        ).reset_index()
        if not best_df.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.plot(best_df["round"], best_df["best_pred"], marker="o", label="Round Best Pred [%]", linewidth=2)
            if "round_best_true" in best_df.columns and best_df["best_true"].notna().any():
                plt.plot(best_df["round"], best_df["best_true"], marker="o", label="Round Best True [%]", linewidth=2)
            plt.xlabel("Round")
            plt.ylabel("Yield [%]")
            plt.title("Best Yield per Round (Predicted vs True)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig.savefig(os.path.join(save_dir, "round_best.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Plot: Acquisition vs yield =====
    if not df_obs.empty and "acquisition_value" in df_obs.columns:
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(df_obs["acquisition_value"], df_obs["actual_yield"], alpha=0.6)
        plt.xlabel("Acquisition Value")
        plt.ylabel("Actual Yield [%]")
        plt.title("Acquisition Value vs Actual Yield")
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "acquisition_vs_yield.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Plot: Optimization progress =====
    if not df_obs.empty:
        # Calculate cumulative maximum
        cumulative_max = np.maximum.accumulate(df_obs["actual_yield"])

        fig = plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(df_obs) + 1), df_obs["actual_yield"].values, 'o-', alpha=0.6, label='Actual Yield',
                 markersize=4)
        plt.plot(range(1, len(df_obs) + 1), cumulative_max, 'r-', linewidth=3, label='Best So Far')
        plt.xlabel('Trial')
        plt.ylabel('Yield [%]')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "optimization_progress.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    print(f"Visualization complete! Saved to: {save_dir}")
    return {"overall": overall, "by_round": mdf, "save_dir": save_dir}


def save_tensorboard_plots(logdir: str, save_dir: str = None, dpi: int = 180):
    """
    Save TensorBoard training curves as images.

    Args:
        logdir: TensorBoard log directory (parent directory containing logs for each round)
        save_dir: Image output directory (if None, uses logdir/tensorboard_visualization)
        dpi: Image resolution

    Returns:
        Path to saved directory
    """
    if save_dir is None:
        save_dir = os.path.join(logdir, "tensorboard_visualization")
    os.makedirs(save_dir, exist_ok=True)

    # Get log directories for each round
    round_dirs = []
    for item in sorted(os.listdir(logdir)):
        item_path = os.path.join(logdir, item)
        if os.path.isdir(item_path) and item.startswith("round_"):
            logs_path = os.path.join(item_path, "logs")
            if os.path.exists(logs_path):
                round_dirs.append((item, logs_path))

    if not round_dirs:
        print(f"No TensorBoard logs found: {logdir}")
        return

    print(f"Detected {len(round_dirs)} round logs")

    # Collect data
    all_train_loss = {}
    all_eval_loss = {}
    all_learning_rate = {}

    for round_name, logs_path in round_dirs:
        try:
            # Find event files
            event_files = [f for f in os.listdir(logs_path) if f.startswith("events.out.tfevents")]
            if not event_files:
                print(f"[Warning] {round_name}: no event files found")
                continue

            event_file = os.path.join(logs_path, event_files[0])
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # Available scalar tags
            scalar_tags = ea.Tags().get('scalars', [])

            # Train Loss
            if 'loss' in scalar_tags or 'train/loss' in scalar_tags:
                tag = 'loss' if 'loss' in scalar_tags else 'train/loss'
                events = ea.Scalars(tag)
                all_train_loss[round_name] = [(e.step, e.value) for e in events]

            # Eval Loss
            if 'eval_loss' in scalar_tags or 'eval/loss' in scalar_tags:
                tag = 'eval_loss' if 'eval_loss' in scalar_tags else 'eval/loss'
                events = ea.Scalars(tag)
                all_eval_loss[round_name] = [(e.step, e.value) for e in events]

            # Learning Rate
            if 'learning_rate' in scalar_tags or 'train/learning_rate' in scalar_tags:
                tag = 'learning_rate' if 'learning_rate' in scalar_tags else 'train/learning_rate'
                events = ea.Scalars(tag)
                all_learning_rate[round_name] = [(e.step, e.value) for e in events]

            print(
                f"[Info] {round_name}: train_loss={len(all_train_loss.get(round_name, []))}, "
                f"eval_loss={len(all_eval_loss.get(round_name, []))}, "
                f"lr={len(all_learning_rate.get(round_name, []))}"
            )

        except Exception as e:
            print(f"[Error] {round_name}: {e}")

    # ===== Plot 1: Training Loss (all rounds) =====
    if all_train_loss:
        fig, ax = plt.subplots(figsize=(10, 6))
        for round_name in sorted(all_train_loss.keys()):
            data = all_train_loss[round_name]
            steps, values = zip(*data)
            ax.plot(steps, values, marker='o', label=round_name, alpha=0.7, markersize=3)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Training Loss")
        ax.set_title("Training Loss Across All Rounds")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "training_loss_all_rounds.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved file: training_loss_all_rounds.png")

    # ===== Plot 2: Validation Loss (all rounds) =====
    if all_eval_loss:
        fig, ax = plt.subplots(figsize=(10, 6))
        for round_name in sorted(all_eval_loss.keys()):
            data = all_eval_loss[round_name]
            steps, values = zip(*data)
            ax.plot(steps, values, marker='o', label=round_name, alpha=0.7, markersize=5, linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Validation Loss Across All Rounds")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "validation_loss_all_rounds.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved file: validation_loss_all_rounds.png")

    # ===== Plot 3: Subplots by round =====
    n_rounds = len(round_dirs)
    if n_rounds > 0:
        n_cols = min(3, n_rounds)
        n_rows = (n_rounds + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rounds == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rounds > 1 else [axes]

        for idx, (round_name, _) in enumerate(sorted(round_dirs)):
            # Train loss
            if round_name in all_train_loss:
                data = all_train_loss[round_name]
                steps, values = zip(*data)
                axes[idx].plot(steps, values, marker='o', label='Train Loss', alpha=0.7, markersize=3)

            # Eval loss
            if round_name in all_eval_loss:
                data = all_eval_loss[round_name]
                steps, values = zip(*data)
                axes[idx].plot(steps, values, marker='s', label='Val Loss', alpha=0.7, markersize=5, linewidth=2)

            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("Loss")
            axes[idx].set_title(round_name)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(round_dirs), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "training_curves_by_round.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved file: training_curves_by_round.png")

    # ===== Plot 4: Learning rate schedule =====
    if all_learning_rate:
        fig, ax = plt.subplots(figsize=(10, 5))
        for round_name in sorted(all_learning_rate.keys()):
            data = all_learning_rate[round_name]
            steps, values = zip(*data)
            ax.plot(steps, values, marker='.', label=round_name, alpha=0.7)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "learning_rate_schedule.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("Saved file: learning_rate_schedule.png")

    print(f"\nTensorBoard plots saved to: {save_dir}")
    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("--csv", type=str, required=True, help="Path to optimization log CSV")
    parser.add_argument("--logdir", type=str, help="TensorBoard log directory (optional)")
    parser.add_argument("--output", type=str, help="Output directory (optional)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--dpi", type=int, default=180, help="Image resolution")

    args = parser.parse_args()

    # Visualize CSV logs
    result = visualize_logs(args.csv, args.output, show=args.show, dpi=args.dpi)
    print(f"\nVisualization saved to: {result['save_dir']}")

    # Visualize TensorBoard logs if provided
    if args.logdir:
        tb_dir = save_tensorboard_plots(args.logdir, args.output, dpi=args.dpi)
        print(f"TensorBoard plots saved to: {tb_dir}")

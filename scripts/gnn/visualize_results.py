"""Visualization utilities for GNN Bayesian optimization results."""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_logs(csv_path: str, out_dir: str = None, show: bool = False, dpi: int = 180):
    """
    Visualize optimization results from CSV log.

    Args:
        csv_path: Path to optimization_log.csv
        out_dir: Output directory for plots (default: same as CSV directory)
        show: Whether to display plots
        dpi: Plot resolution
    """
    # Output directory
    root = out_dir or os.path.dirname(csv_path) or "."
    save_dir = os.path.join(root, "visualization")
    os.makedirs(save_dir, exist_ok=True)

    # Load and format data
    df = pd.read_csv(csv_path)
    to_num_cols = [
        "trial", "index", "predicted_mean", "predicted_std", "actual_yield",
        "error_pct", "acquisition_value", "cumulative_data_size", "best_so_far"
    ]
    for c in to_num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Data with actual yield
    df_obs = df.dropna(subset=["actual_yield"]) if "actual_yield" in df.columns else pd.DataFrame()

    # ===== Overall metrics =====
    overall = {}
    if not df_obs.empty and "predicted_mean" in df_obs.columns:
        pred_data = df_obs.dropna(subset=["predicted_mean"])
        if not pred_data.empty:
            y = pred_data["actual_yield"].to_numpy()
            yhat = pred_data["predicted_mean"].to_numpy()
            err = yhat - y
            overall = {
                "n": int(len(pred_data)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "bias": float(np.mean(err)),
            }
    pd.DataFrame([overall]).to_csv(os.path.join(save_dir, "metrics_overall.csv"), index=False)

    # ===== Figure: Parity plot =====
    if not df_obs.empty and "predicted_mean" in df_obs.columns:
        pred_data = df_obs.dropna(subset=["predicted_mean"])
        if not pred_data.empty:
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(pred_data["actual_yield"], pred_data["predicted_mean"], s=18, alpha=0.65)
            lims = [0, 100]
            plt.plot(lims, lims, linestyle="--", color="red", label="Perfect Prediction")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.xlabel("True Yield (%)")
            plt.ylabel("Predicted Yield (%)")
            plt.title("Parity: Prediction vs Truth")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "parity.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Figure: Error histogram =====
    if not df_obs.empty and "error_pct" in df_obs.columns:
        error_data = df_obs["error_pct"].dropna()
        if not error_data.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.hist(error_data.to_numpy(), bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
            plt.xlabel("Prediction Error (%) (pred - true)")
            plt.ylabel("Count")
            plt.title("Error Histogram")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "error_hist.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Figure: Uncertainty vs actual yield =====
    if not df_obs.empty and "predicted_std" in df_obs.columns:
        uncertainty_data = df_obs.dropna(subset=["predicted_std"])
        if not uncertainty_data.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(uncertainty_data["predicted_std"], uncertainty_data["actual_yield"], alpha=0.6)
            plt.xlabel("Prediction Uncertainty (Std) (%)")
            plt.ylabel("Actual Yield (%)")
            plt.title("Uncertainty vs Actual Yield")
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "uncertainty_vs_yield.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Figure: Acquisition value vs actual yield =====
    if not df_obs.empty and "acquisition_value" in df_obs.columns:
        acq_data = df_obs.dropna(subset=["acquisition_value"])
        if not acq_data.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(acq_data["acquisition_value"], acq_data["actual_yield"], alpha=0.6)
            plt.xlabel("Acquisition Value")
            plt.ylabel("Actual Yield (%)")
            plt.title("Acquisition Value vs Actual Yield")
            plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(save_dir, "acquisition_vs_yield.png"), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

    # ===== Figure: Optimization progress =====
    if not df_obs.empty:
        fig = plt.figure(figsize=(8, 4))
        cumulative_max = np.maximum.accumulate(df_obs["actual_yield"])

        plt.plot(range(1, len(df_obs) + 1), df_obs["actual_yield"].values, 'o-', alpha=0.6,
                 label='Actual Yield', markersize=4)
        plt.plot(range(1, len(df_obs) + 1), cumulative_max, 'r-', linewidth=3, label='Best So Far')
        plt.xlabel('Trial')
        plt.ylabel('Yield (%)')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "optimization_progress.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Figure: Best yield per selection type =====
    if not df_obs.empty and "selection_type" in df_obs.columns:
        fig = plt.figure(figsize=(6, 4))
        selection_types = df_obs["selection_type"].unique()
        colors = {'random': 'blue', 'bayesian': 'orange'}

        for sel_type in selection_types:
            type_data = df_obs[df_obs["selection_type"] == sel_type]
            if not type_data.empty:
                cumulative_max = np.maximum.accumulate(type_data["actual_yield"])
                iterations = type_data["trial"].values
                plt.plot(iterations, cumulative_max, '-', linewidth=2,
                         label=f'{sel_type.capitalize()}', color=colors.get(sel_type, 'gray'))

        plt.xlabel('Trial')
        plt.ylabel('Best Yield So Far (%)')
        plt.title('Best Yield by Selection Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "best_by_selection_type.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ===== Figure: Cumulative data size =====
    if not df_obs.empty and "cumulative_data_size" in df_obs.columns:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(df_obs["trial"], df_obs["cumulative_data_size"], 'o-', linewidth=2)
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Training Data Size')
        plt.title('Training Data Growth')
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(save_dir, "data_size_growth.png"), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    print(f"Visualization completed! Saved to: {save_dir}")
    return {"overall": overall, "save_dir": save_dir}


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN Bayesian optimization results")
    parser.add_argument("--csv", type=str, required=True, help="Path to optimization_log.csv")
    parser.add_argument("--logdir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots")
    parser.add_argument("--dpi", type=int, default=180, help="Plot resolution")

    args = parser.parse_args()

    result = visualize_logs(
        csv_path=args.csv,
        out_dir=args.logdir,
        show=args.show,
        dpi=args.dpi
    )

    print("\nVisualization completed successfully!")
    print(f"Results saved to: {result['save_dir']}")


if __name__ == "__main__":
    main()

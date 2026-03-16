"""Evaluation metrics and plotting."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    log_loss, matthews_corrcoef, cohen_kappa_score, roc_curve, precision_recall_curve,
)

METRICS = ['accuracy', 'balanced_accuracy', 'f1_macro', 'auroc_macro',
           'auprc_macro', 'mcc', 'precision_macro', 'recall_macro']


# =============================================================================
# Curve Fitting
# =============================================================================

def fit_curve(x: np.ndarray, y: np.ndarray, func_type: str = "log") -> Tuple[callable, np.ndarray, float, str]:
    """Fit a curve to data points. Returns (func, params, r_squared, equation_string)."""
    funcs = {
        "log": (lambda x, a, b: a * np.log(x) + b, [0.1, 0.5]),
        "linear": (lambda x, a, b: a * x + b, [0.001, 0.5]),
        "power": (lambda x, a, b, c: a * np.power(x, b) + c, [0.5, 0.3, 0.5]),
    }
    func, p0 = funcs[func_type]
    try:
        params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
        y_pred = func(x, *params)
        ss_res, ss_tot = np.sum((y - y_pred) ** 2), np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        if func_type == "log":
            eq = f"y = {params[0]:.4f}·log(x) + {params[1]:.4f}"
        elif func_type == "linear":
            eq = f"y = {params[0]:.6f}x + {params[1]:.4f}"
        else:
            eq = f"y = {params[0]:.4f}·x^{params[1]:.4f} + {params[2]:.4f}"
        return func, params, r2, eq
    except Exception:
        return None, None, 0, ""


def save_curve_fits(curve_fits: Dict, output_path: str) -> None:
    """Save curve fit parameters to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays and types to native Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(curve_fits)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Curve fits saved to {output_path}")


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    num_classes: int = 10) -> Dict:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth labels (should be 0 to num_classes-1)
        y_pred: Predicted labels
        y_prob: Prediction probabilities, shape (n_samples, num_classes)
        num_classes: Number of classes
    """
    # Ensure labels are in expected range [0, num_classes-1]
    labels = list(range(num_classes))

    # Top-k accuracy (adjust k based on num_classes)
    top_3_k = min(3, num_classes)
    top_5_k = min(5, num_classes)
    top_k = lambda k: np.mean([y_true[i] in np.argsort(y_prob[i])[-k:] for i in range(len(y_true))])

    # AUPRC macro (per-class average)
    auprc_macro = np.mean([average_precision_score((y_true == i).astype(int), y_prob[:, i])
                           for i in range(num_classes)])

    # Handle AUROC for binary vs multi-class
    if num_classes == 2:
        # Binary classification: use probability of positive class
        auroc_macro = roc_auc_score(y_true, y_prob[:, 1])
        auroc_weighted = auroc_macro  # Same for binary
    else:
        # Multi-class: use OVR
        auroc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
        auroc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted", labels=labels)

    global_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "top_3_accuracy": top_k(top_3_k),
        "top_5_accuracy": top_k(top_5_k),
        "log_loss": log_loss(y_true, y_prob, labels=labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohens_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0, labels=labels),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0, labels=labels),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0, labels=labels),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels),
        "auroc_macro": auroc_macro,
        "auroc_weighted": auroc_weighted,
        "auprc_macro": auprc_macro,
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i in range(num_classes):
        tp, fn = cm[i, i], cm[i, :].sum() - cm[i, i]
        fp, tn = cm[:, i].sum() - cm[i, i], cm.sum() - cm[i, i] - fn - (cm[:, i].sum() - cm[i, i])
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        y_bin = (y_true == i).astype(int)
        per_class[str(i)] = {
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "specificity": tn / (tn + fp) if tn + fp else 0,
            "fpr": fp / (fp + tn) if fp + tn else 0, "fnr": fn / (fn + tp) if fn + tp else 0,
            "auroc": roc_auc_score(y_bin, y_prob[:, i]),
            "auprc": average_precision_score(y_bin, y_prob[:, i]),
            "support": int((y_true == i).sum()),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    return {"global_metrics": global_metrics, "per_class_metrics": per_class, "confusion_matrix": cm.tolist()}


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   num_classes: int = 10) -> Dict:
    """Evaluate model and compute all metrics."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_labels.append(y.numpy())
            all_preds.append(out.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(out, 1).cpu().numpy())
    return compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds),
                          np.concatenate(all_probs), num_classes)


# =============================================================================
# Results Loading
# =============================================================================

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all results from a results directory into a DataFrame."""
    results_path = Path(results_dir)
    agg_file = results_path / "aggregated" / "all_results.json"

    if agg_file.exists():
        with open(agg_file) as f:
            data = json.load(f)
        rows = [{"seed": r["seed"], **r["condition"], **r["training_dynamics"], **r["global_metrics"]}
                for r in data]
    else:
        rows = []
        for seed_dir in (results_path / "raw").iterdir():
            if seed_dir.is_dir():
                for json_file in seed_dir.glob("*.json"):
                    with open(json_file) as f:
                        r = json.load(f)
                    rows.append({"seed": r["seed"], **r["condition"], **r["training_dynamics"], **r["global_metrics"]})
    return pd.DataFrame(rows)


def aggregate_by_condition(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate results by condition, computing mean and std across seeds."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['seed', group_col]]
    agg = df.groupby(group_col)[numeric_cols].agg(['mean', 'std'])
    agg.columns = ['_'.join(col) for col in agg.columns]
    return agg.reset_index()


# =============================================================================
# Plotting Helpers
# =============================================================================

def _add_fit(ax, x, y, fit_type, color, label_prefix=""):
    """Add curve fit to axis if possible. Returns (equation_string, r2, func_type, params)."""
    if fit_type and len(x) > 2:
        func, params, r2, eq = fit_curve(x, y, fit_type)
        if func is not None:
            x_smooth = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_smooth, func(x_smooth, *params), '--', color=color, linewidth=1.5, alpha=0.7,
                   label=f'{label_prefix} R²={r2:.3f}' if label_prefix else f'R²={r2:.4f}')
            return eq, r2, fit_type, params.tolist()
    return None, None, None, None


def _save_fig(fig, output_path):
    """Save figure if path provided."""
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_metrics_grid(df: pd.DataFrame, x_col: str, metrics: List[str], title: str,
                      xlabel: str, output_path: Optional[str] = None, fit_type: str = "log",
                      curve_fits: Optional[Dict] = None) -> plt.Figure:
    """Plot grid of metrics vs x_col with curve fitting."""
    agg = aggregate_by_condition(df, x_col)
    n, cols = len(metrics), min(4, len(metrics))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col in agg.columns:
            x, y = agg[x_col].values, agg[mean_col].values
            yerr = agg[std_col].values if std_col in agg.columns else None
            ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, label='Data', zorder=5)
            eq, r2, ftype, params = _add_fit(ax, x, y, fit_type, 'red')
            if eq:
                ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                # Store curve fit parameters
                if curve_fits is not None:
                    curve_fits[metric] = {
                        "func_type": ftype,
                        "params": params,
                        "r_squared": r2,
                        "equation": eq,
                        "x_min": float(x.min()),
                        "x_max": float(x.max()),
                        "x_col": x_col
                    }
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return _save_fig(fig, output_path)


def plot_marginal_gains(df: pd.DataFrame, x_col: str, metric: str, title: str,
                        output_path: Optional[str] = None) -> plt.Figure:
    """Plot marginal gains per increment."""
    agg = aggregate_by_condition(df, x_col)
    x, y = agg[x_col].values, agg[f"{metric}_mean"].values
    gains = np.diff(y)
    labels = [f"{x[i]}→{x[i+1]}" for i in range(len(gains))]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['green' if g > 0 else 'red' for g in gains]
    ax.bar(range(len(gains)), gains, color=colors, alpha=0.7)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel(f'{x_col} Increment')
    ax.set_ylabel(f'Δ {metric}')
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    return _save_fig(fig, output_path)


def plot_exp2_comparison(df: pd.DataFrame, x_col: str, metric: str, baseline: Optional[Dict],
                         title: str, xlabel: str, ylabel: str, output_path: Optional[str] = None,
                         curve_fits: Optional[Dict] = None) -> plt.Figure:
    """Plot exp2 comparison with noise rates + baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Dynamic color palette: black for baseline, then colormap for noise rates
    noise_rates = sorted(df['noise_rate'].unique())
    n_noise = len(noise_rates)
    cmap = plt.colormaps['tab10']
    noise_colors = [cmap(i) for i in range(n_noise)]

    equations = []

    # Plot baseline (exp1 clean data) - use LOG fit since it spans full range
    if baseline and 'x' in baseline and 'y' in baseline:
        bx, by, byerr = baseline['x'], baseline['y'], baseline.get('yerr')
        ax.errorbar(bx, by, yerr=byerr, marker='s', capsize=3, color='black',
                   label='0% noise (clean)', zorder=5)
        eq, r2, ftype, params = _add_fit(ax, bx, by, 'log', 'black', '0% ')
        if eq:
            equations.append(f"0%: {eq}")
            # Store baseline curve fit
            if curve_fits is not None:
                curve_fits[f"{metric}_baseline"] = {
                    "func_type": ftype,
                    "params": params,
                    "r_squared": r2,
                    "equation": eq,
                    "x_min": float(bx.min()),
                    "x_max": float(bx.max()),
                    "x_col": x_col,
                    "noise_rate": 0.0
                }

    # Plot noise rate lines - use LINEAR fit
    for i, gval in enumerate(noise_rates):
        subset = df[df['noise_rate'] == gval]
        agg = aggregate_by_condition(subset, x_col)
        x, y = agg[x_col].values, agg[f"{metric}_mean"].values
        yerr = agg.get(f"{metric}_std")
        yerr = yerr.values if yerr is not None else None

        noise_label = f"{int(gval * 100)}%"
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, color=noise_colors[i],
                   label=f'{noise_label} noise', zorder=5)
        eq, r2, ftype, params = _add_fit(ax, x, y, 'linear', noise_colors[i], f'{noise_label} ')
        if eq:
            equations.append(f"{noise_label}: {eq}")
            # Store noise rate curve fit
            if curve_fits is not None:
                curve_fits[f"{metric}_noise_{int(gval*100)}pct"] = {
                    "func_type": ftype,
                    "params": params,
                    "r_squared": r2,
                    "equation": eq,
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                    "x_col": x_col,
                    "noise_rate": gval
                }

    # Add equations text box (top right, outside main data area)
    if equations:
        eq_text = '\n'.join(equations)
        ax.text(0.98, 0.02, eq_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix",
                          output_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    return _save_fig(fig, output_path)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, title: str = "ROC Curves",
                    output_path: Optional[str] = None) -> plt.Figure:
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(y_prob.shape[1]):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc_score(y_bin, y_prob[:, i]):.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save_fig(fig, output_path)


# =============================================================================
# Plot Generation
# =============================================================================

def generate_exp1_plots(results_dir: str, output_dir: str) -> Dict:
    """Generate all plots for Experiment 1. Returns curve fits dictionary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)

    # Dictionary to store all curve fits
    curve_fits = {"experiment": "exp1", "fits": {}}

    # Grid plots with curve fitting - store fits for absolute (num_samples) plots
    plot_metrics_grid(df, 'subset_pct', METRICS, 'Experiment 1: Metrics vs Data %',
                     'Training Data (%)', f'{output_dir}/learning_curves_pct.png')

    # Store curve fits from absolute plots (num_samples)
    plot_metrics_grid(df, 'num_samples', METRICS, 'Experiment 1: Metrics vs Sample Count',
                     'Number of Samples', f'{output_dir}/learning_curves_absolute.png',
                     curve_fits=curve_fits["fits"])

    # Marginal gains
    plot_marginal_gains(df, 'subset_pct', 'accuracy', 'Marginal Accuracy Gains (by %)',
                       f'{output_dir}/marginal_gains_pct.png')
    plot_marginal_gains(df, 'num_samples', 'accuracy', 'Marginal Accuracy Gains (by Sample Count)',
                       f'{output_dir}/marginal_gains_absolute.png')

    # Individual metrics
    for m in METRICS:
        plot_metrics_grid(df, 'subset_pct', [m], f'{m.replace("_", " ").title()} vs Data %',
                         'Training Data (%)', f'{output_dir}/{m}_pct.png')
        plot_metrics_grid(df, 'num_samples', [m], f'{m.replace("_", " ").title()} vs Sample Count',
                         'Number of Samples', f'{output_dir}/{m}_absolute.png')

    # Save curve fits to JSON (in aggregated folder, not figures)
    agg_dir = str(Path(output_dir).parent / "aggregated")
    save_curve_fits(curve_fits, f'{agg_dir}/curve_fits.json')

    print(f"Exp1 plots saved to {output_dir}")
    return curve_fits


def generate_exp2_plots(results_dir: str, output_dir: str, exp1_dir: Optional[str] = None) -> Dict:
    """Generate all plots for Experiment 2. Returns curve fits dictionary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)

    # Dictionary to store all curve fits
    curve_fits = {"experiment": "exp2", "fits": {}}

    # Load exp1 baseline (full range for proper log fit on absolute plots)
    baselines_abs = {}
    if exp1_dir:
        try:
            exp1_df = load_results(exp1_dir)
            if not exp1_df.empty:
                exp1_agg = aggregate_by_condition(exp1_df, 'num_samples')
                for m in METRICS:
                    mean_col, std_col = f"{m}_mean", f"{m}_std"
                    if mean_col in exp1_agg.columns:
                        baselines_abs[m] = {
                            'x': exp1_agg['num_samples'].values,
                            'y': exp1_agg[mean_col].values,
                            'yerr': exp1_agg.get(std_col, pd.Series()).values if std_col in exp1_agg.columns else None
                        }
                print(f"Loaded exp1 baseline (full range)")
        except Exception as e:
            print(f"Could not load exp1: {e}")

    for m in METRICS:
        # Percentage plot - NO baseline (x-axes aren't comparable)
        plot_exp2_comparison(df, 'noisy_pct', m, None,
                            f'{m.replace("_", " ").title()} vs Noisy Data %',
                            'Noisy Data (%)', m.replace('_', ' ').title(),
                            f'{output_dir}/{m}_pct.png')
        # Absolute plot - WITH baseline (x = total samples, comparable)
        plot_exp2_comparison(df, 'total_samples', m, baselines_abs.get(m),
                            f'{m.replace("_", " ").title()} vs Total Samples',
                            'Total Samples', m.replace('_', ' ').title(),
                            f'{output_dir}/{m}_absolute.png',
                            curve_fits=curve_fits["fits"])

    # Save curve fits to JSON (in aggregated folder, not figures)
    agg_dir = str(Path(output_dir).parent / "aggregated")
    save_curve_fits(curve_fits, f'{agg_dir}/curve_fits.json')

    print(f"Exp2 plots saved to {output_dir}")
    return curve_fits


# =============================================================================
# Experiment 3 Plotting
# =============================================================================

def load_exp3_results(results_dir: str) -> pd.DataFrame:
    """Load exp3 results from aggregated JSON or raw files."""
    results_path = Path(results_dir)

    # Try to find aggregated results (with any n prefix)
    agg_dir = results_path / "aggregated"
    if agg_dir.exists():
        json_files = list(agg_dir.glob("all_results_n*.json"))
        if json_files:
            # Use the most recent one
            agg_file = sorted(json_files)[-1]
            with open(agg_file) as f:
                data = json.load(f)
            rows = []
            for r in data:
                rows.append({
                    "seed": r["seed"],
                    **r["condition"],
                    **r["training_dynamics"],
                    **r["global_metrics"]
                })
            return pd.DataFrame(rows)

    # Fallback to raw files
    rows = []
    raw_dir = results_path / "raw"
    if raw_dir.exists():
        for seed_dir in raw_dir.iterdir():
            if seed_dir.is_dir():
                for json_file in seed_dir.glob("*.json"):
                    with open(json_file) as f:
                        r = json.load(f)
                    rows.append({
                        "seed": r["seed"],
                        **r["condition"],
                        **r["training_dynamics"],
                        **r["global_metrics"]
                    })

    return pd.DataFrame(rows)


def plot_exp3_learning_curves(df: pd.DataFrame, metric: str, title: str,
                               output_path: Optional[str] = None,
                               curve_fits: Optional[Dict] = None) -> plt.Figure:
    """Plot learning curves for each class count (9 lines on one plot)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    class_counts = sorted(df['num_classes'].unique())
    cmap = plt.colormaps['tab10']
    equations = []

    for i, nc in enumerate(class_counts):
        subset = df[df['num_classes'] == nc]
        agg = aggregate_by_condition(subset, 'subset_pct')

        x = agg['subset_pct'].values
        y = agg[f'{metric}_mean'].values
        yerr = agg.get(f'{metric}_std')
        yerr = yerr.values if yerr is not None else None

        color = cmap(i / len(class_counts))
        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, color=color,
                   label=f'{nc} classes', linewidth=2, markersize=6)

        # Fit curve and store
        if len(x) > 2:
            func, params, r2, eq = fit_curve(x, y, "log")
            if func is not None:
                x_smooth = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_smooth, func(x_smooth, *params), '--', color=color,
                       linewidth=1, alpha=0.5)
                equations.append(f"{nc}c: R²={r2:.3f}")
                if curve_fits is not None:
                    curve_fits[f"{metric}_classes_{nc}"] = {
                        "func_type": "log",
                        "params": params.tolist(),
                        "r_squared": r2,
                        "equation": eq,
                        "x_min": float(x.min()),
                        "x_max": float(x.max()),
                        "x_col": "subset_pct",
                        "num_classes": int(nc)
                    }

    # Add equations text box
    if equations:
        eq_text = '\n'.join(equations)
        ax.text(0.98, 0.02, eq_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               family='monospace')

    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, output_path)


def plot_exp3_heatmap(df: pd.DataFrame, metric: str, title: str,
                       output_path: Optional[str] = None) -> plt.Figure:
    """Plot heatmap: num_classes × data_pct → metric."""
    # Aggregate by (num_classes, subset_pct)
    pivot_data = df.groupby(['num_classes', 'subset_pct'])[metric].mean().reset_index()
    pivot_table = pivot_data.pivot(index='num_classes', columns='subset_pct', values=metric)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})

    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel('Number of Classes', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return _save_fig(fig, output_path)


def plot_exp3_iso_accuracy(df: pd.DataFrame, target_accuracies: List[float],
                            output_path: Optional[str] = None,
                            curve_fits: Optional[Dict] = None) -> plt.Figure:
    """Plot iso-accuracy curves: data needed to reach target accuracy vs num_classes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    class_counts = sorted(df['num_classes'].unique())
    colors = plt.colormaps['viridis'](np.linspace(0, 0.8, len(target_accuracies)))

    for target_acc, color in zip(target_accuracies, colors):
        data_needed = []
        valid_classes = []

        for nc in class_counts:
            subset = df[df['num_classes'] == nc]
            agg = aggregate_by_condition(subset, 'subset_pct')

            x = agg['subset_pct'].values
            y = agg['accuracy_mean'].values

            # Find where accuracy crosses target
            if y.max() >= target_acc:
                # Interpolate to find exact crossing point
                idx = np.where(y >= target_acc)[0]
                if len(idx) > 0:
                    first_idx = idx[0]
                    if first_idx == 0:
                        data_needed.append(x[0])
                    else:
                        # Linear interpolation
                        x0, x1 = x[first_idx-1], x[first_idx]
                        y0, y1 = y[first_idx-1], y[first_idx]
                        x_interp = x0 + (target_acc - y0) * (x1 - x0) / (y1 - y0)
                        data_needed.append(x_interp)
                    valid_classes.append(nc)
            else:
                # Target not reached
                pass

        if valid_classes:
            ax.plot(valid_classes, data_needed, 'o-', color=color, linewidth=2,
                   markersize=8, label=f'{int(target_acc*100)}% accuracy')

            # Fit scaling law if enough points
            if len(valid_classes) >= 3:
                x_arr = np.array(valid_classes)
                y_arr = np.array(data_needed)
                func, params, r2, eq = fit_curve(x_arr, y_arr, "power")
                if func is not None:
                    x_smooth = np.linspace(min(valid_classes), max(valid_classes), 100)
                    ax.plot(x_smooth, func(x_smooth, *params), '--', color=color,
                           linewidth=1, alpha=0.7)
                    # Add equation text
                    ax.text(0.02, 0.98 - 0.06 * target_accuracies.index(target_acc),
                           f'{int(target_acc*100)}%: {eq} (R²={r2:.3f})',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    if curve_fits is not None:
                        curve_fits[f"iso_accuracy_{int(target_acc*100)}pct"] = {
                            "func_type": "power",
                            "params": params.tolist(),
                            "r_squared": r2,
                            "equation": eq,
                            "target_accuracy": float(target_acc),
                            "x_min": float(min(valid_classes)),
                            "x_max": float(max(valid_classes))
                        }

    ax.set_xlabel('Number of Classes', fontsize=12)
    ax.set_ylabel('Training Data (%) Needed', fontsize=12)
    ax.set_title('Data Required to Reach Target Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(class_counts)

    return _save_fig(fig, output_path)


def plot_exp3_scaling_law(df: pd.DataFrame, target_pct: int,
                           output_path: Optional[str] = None,
                           curve_fits: Optional[Dict] = None) -> plt.Figure:
    """Plot accuracy vs num_classes at a fixed data percentage to show scaling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = df[df['subset_pct'] == target_pct]
    if subset.empty:
        # Find closest available percentage
        available_pcts = sorted(df['subset_pct'].unique())
        target_pct = min(available_pcts, key=lambda x: abs(x - target_pct))
        subset = df[df['subset_pct'] == target_pct]

    agg = aggregate_by_condition(subset, 'num_classes')

    x = agg['num_classes'].values
    y = agg['accuracy_mean'].values
    yerr = agg.get('accuracy_std')
    yerr = yerr.values if yerr is not None else None

    ax.errorbar(x, y, yerr=yerr, marker='o', capsize=3, color='blue',
               linewidth=2, markersize=8, label='Observed')

    # Fit curve
    if len(x) > 2:
        func, params, r2, eq = fit_curve(x, y, "log")
        if func is not None:
            x_smooth = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_smooth, func(x_smooth, *params), '--', color='red',
                   linewidth=2, alpha=0.7, label=f'Fit (R²={r2:.4f})')
            # Add equation text box
            ax.text(0.02, 0.02, f'{eq}\nR² = {r2:.4f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            if curve_fits is not None:
                curve_fits[f"scaling_law_pct_{target_pct}"] = {
                    "func_type": "log",
                    "params": params.tolist(),
                    "r_squared": r2,
                    "equation": eq,
                    "data_pct": int(target_pct),
                    "x_min": float(x.min()),
                    "x_max": float(x.max())
                }

    ax.set_xlabel('Number of Classes', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs Number of Classes (at {target_pct}% data)',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    return _save_fig(fig, output_path)


def plot_exp3_class_difficulty(df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """Plot which class combinations are hardest/easiest."""
    # Get accuracy at 100% data for each combination
    full_data = df[df['subset_pct'] == df['subset_pct'].max()].copy()

    # Convert class_list to string for grouping (it may be a list)
    full_data['class_list_str'] = full_data['class_list'].apply(
        lambda x: str(x) if isinstance(x, list) else x
    )

    # Group by class_list_str and num_classes
    combo_perf = full_data.groupby(['num_classes', 'class_list_str']).agg({
        'accuracy': ['mean', 'std']
    }).reset_index()
    combo_perf.columns = ['num_classes', 'class_list', 'accuracy_mean', 'accuracy_std']

    fig, ax = plt.subplots(figsize=(12, 6))

    class_counts = sorted(combo_perf['num_classes'].unique())
    positions = []
    labels = []
    colors = []
    cmap = plt.colormaps['tab10']

    pos = 0
    for nc in class_counts:
        nc_data = combo_perf[combo_perf['num_classes'] == nc].sort_values('accuracy_mean')
        for _, row in nc_data.iterrows():
            positions.append(pos)
            labels.append(f"{nc}c: {row['class_list']}")
            colors.append(cmap(nc / 10))
            ax.barh(pos, row['accuracy_mean'], xerr=row['accuracy_std'],
                   color=cmap(nc / 10), alpha=0.7, capsize=3)
            pos += 1
        pos += 0.5  # Gap between class counts

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy by Class Combination (at max data %)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    return _save_fig(fig, output_path)


def generate_exp3_plots(results_dir: str, output_dir: str,
                         target_accuracies: Optional[List[float]] = None) -> Dict:
    """Generate all plots for Experiment 3."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading exp3 results...")
    df = load_exp3_results(results_dir)

    if df.empty:
        print("No exp3 results found!")
        return {}

    print(f"Loaded {len(df)} results")

    # Default target accuracies
    if target_accuracies is None:
        target_accuracies = [0.90]

    # Dictionary to store curve fits
    curve_fits = {"experiment": "exp3", "fits": {}}

    # 1. Learning curves by class count (one plot per metric)
    for m in METRICS:
        plot_exp3_learning_curves(df, m, f'{m.replace("_", " ").title()} vs Data % by Class Count',
                                   f'{output_dir}/{m}_by_class_count.png',
                                   curve_fits["fits"])

    # 2. Heatmaps
    for m in ['accuracy', 'f1_macro', 'auroc_macro']:
        plot_exp3_heatmap(df, m, f'{m.replace("_", " ").title()}: Classes × Data %',
                          f'{output_dir}/{m}_heatmap.png')

    # 3. Iso-accuracy curves
    plot_exp3_iso_accuracy(df, target_accuracies, f'{output_dir}/iso_accuracy_curves.png',
                            curve_fits["fits"])

    # 4. Scaling law at different data percentages
    for pct in [50, 100]:
        if pct in df['subset_pct'].values:
            plot_exp3_scaling_law(df, pct, f'{output_dir}/scaling_law_pct_{pct}.png',
                                   curve_fits["fits"])

    # 5. Class difficulty comparison
    plot_exp3_class_difficulty(df, f'{output_dir}/class_difficulty.png')

    # Save curve fits
    agg_dir = str(Path(output_dir).parent / "aggregated")
    Path(agg_dir).mkdir(parents=True, exist_ok=True)
    save_curve_fits(curve_fits, f'{agg_dir}/curve_fits.json')

    print(f"Exp3 plots saved to {output_dir}")
    return curve_fits


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--exp1", nargs="?", const="results/exp1", help="Generate exp1 plots")
    parser.add_argument("--exp2", nargs="?", const="results/exp2", help="Generate exp2 plots")
    parser.add_argument("--exp3", nargs="?", const="results/exp3", help="Generate exp3 plots")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--exp1-path", default="results/exp1", help="Exp1 results path")
    parser.add_argument("--exp2-path", default="results/exp2", help="Exp2 results path")
    parser.add_argument("--exp3-path", default="results/exp3", help="Exp3 results path")
    args = parser.parse_args()

    if not any([args.exp1, args.exp2, args.exp3, args.all]):
        args.all = True

    exp1_path = args.exp1 or args.exp1_path
    exp2_path = args.exp2 or args.exp2_path
    exp3_path = args.exp3 or args.exp3_path

    if args.exp1 or args.all:
        print(f"\n{'='*60}\nGenerating Exp1 plots: {exp1_path}\n{'='*60}")
        try:
            generate_exp1_plots(exp1_path, f"{exp1_path}/figures")
        except Exception as e:
            print(f"Error: {e}")

    if args.exp2 or args.all:
        print(f"\n{'='*60}\nGenerating Exp2 plots: {exp2_path}\n{'='*60}")
        try:
            generate_exp2_plots(exp2_path, f"{exp2_path}/figures", exp1_path)
        except Exception as e:
            print(f"Error: {e}")

    if args.exp3 or args.all:
        print(f"\n{'='*60}\nGenerating Exp3 plots: {exp3_path}\n{'='*60}")
        try:
            generate_exp3_plots(exp3_path, f"{exp3_path}/figures")
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == "__main__":
    main()
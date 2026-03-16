#!/usr/bin/env python3
"""Experiment 3: Optimal Data per Number of Classes.

This experiment determines how the optimal amount of training data scales
with the number of classes being classified.

Approach: Multiple Random Samplings
- For each class count (2-10), randomly sample N different class combinations
- For each combination, vary training data percentage
- Average results across combinations to reduce selection bias

Usage:
    python exp3_class_scaling.py
    python exp3_class_scaling.py --output results/exp3 --quiet
"""

import argparse
import csv
import gc
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import CONFIG, DEVICE, CLASS_NAMES
from utils import set_seed, save_json
from model import CNN
from trainer import train_model
from evaluate import compute_metrics, generate_exp3_plots


def cleanup_memory():
    """Clean up GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Dataset with Class Subset Support
# =============================================================================

class FashionMNISTClassSubset:
    """Fashion-MNIST with support for class subsetting and stratified sampling."""

    def __init__(self, data_dir: str = "./data"):
        self.transform = transforms.Normalize((0.2860,), (0.3530,))
        train = datasets.FashionMNIST(data_dir, train=True, download=True)
        test = datasets.FashionMNIST(data_dir, train=False, download=True)
        self.train_data = train.data.numpy()
        self.train_labels = train.targets.numpy()
        self.test_data = test.data.numpy()
        self.test_labels = test.targets.numpy()
        self.num_classes = 10

    def get_class_indices(self, class_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices for samples belonging to specified classes."""
        train_mask = np.isin(self.train_labels, class_list)
        test_mask = np.isin(self.test_labels, class_list)
        return np.where(train_mask)[0], np.where(test_mask)[0]

    def remap_labels(self, labels: np.ndarray, class_list: List[int]) -> np.ndarray:
        """Remap original labels to 0, 1, 2, ... based on class_list order."""
        label_map = {orig: new for new, orig in enumerate(class_list)}
        return np.array([label_map[l] for l in labels])

    def get_nested_indices_for_classes(self, class_list: List[int],
                                       fractions: List[float],
                                       seed: int) -> Dict[float, np.ndarray]:
        """Get nested stratified indices for a subset of classes."""
        train_indices, _ = self.get_class_indices(class_list)
        subset_labels = self.train_labels[train_indices]

        indices_map = {}
        sorted_fracs = sorted(fractions)
        all_idx = np.arange(len(train_indices))

        # Get smallest fraction first
        if sorted_fracs[0] >= 1.0:
            current = all_idx
        else:
            _, current = train_test_split(all_idx, test_size=sorted_fracs[0],
                                          stratify=subset_labels, random_state=seed)
        current = np.sort(current)
        indices_map[sorted_fracs[0]] = train_indices[current]
        remaining = np.setdiff1d(all_idx, current)

        for i, frac in enumerate(sorted_fracs[1:], 1):
            if frac >= 1.0:
                indices_map[frac] = train_indices
                continue
            target_size = int(len(train_indices) * frac)
            needed = target_size - len(current)
            if needed > 0 and len(remaining) > 0:
                rem_labels = subset_labels[remaining]
                if needed >= len(remaining):
                    additional = remaining
                else:
                    _, additional = train_test_split(remaining, test_size=needed / len(remaining),
                                                     stratify=rem_labels, random_state=seed + i)
                current = np.sort(np.concatenate([current, additional]))
                remaining = np.setdiff1d(all_idx, current)
            indices_map[frac] = train_indices[current]

        return indices_map

    def create_loaders_for_classes(self, indices: np.ndarray, class_list: List[int],
                                   batch_size: int, val_split: float,
                                   seed: int) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders for a class subset."""
        data = self.train_data[indices]
        labels = self.remap_labels(self.train_labels[indices], class_list)

        train_idx, val_idx = train_test_split(np.arange(len(indices)), test_size=val_split,
                                              stratify=labels, random_state=seed)

        def make_loader(idx, shuffle):
            x = torch.tensor(data[idx], dtype=torch.float32).unsqueeze(1) / 255.0
            x = self.transform(x)
            y = torch.tensor(labels[idx], dtype=torch.long)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

        return make_loader(train_idx, True), make_loader(val_idx, False)

    def get_test_loader_for_classes(self, class_list: List[int],
                                    batch_size: int) -> DataLoader:
        """Create test dataloader for a class subset."""
        _, test_indices = self.get_class_indices(class_list)
        data = self.test_data[test_indices]
        labels = self.remap_labels(self.test_labels[test_indices], class_list)

        x = torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0
        x = self.transform(x)
        y = torch.tensor(labels, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# =============================================================================
# Class Combination Sampling
# =============================================================================

def sample_class_combinations(num_classes: int, num_samples: int,
                              total_classes: int = 10, seed: int = 42) -> List[Tuple[int, ...]]:
    """
    Sample random class combinations for a given class count.

    Args:
        num_classes: Number of classes to select
        num_samples: Number of different combinations to sample
        total_classes: Total number of available classes (default 10 for Fashion-MNIST)
        seed: Random seed for reproducibility

    Returns:
        List of tuples, each containing class indices
    """
    rng = np.random.RandomState(seed)
    all_combinations = list(combinations(range(total_classes), num_classes))

    if num_samples >= len(all_combinations):
        # Return all combinations if we want more than available
        return all_combinations

    # Randomly sample without replacement
    indices = rng.choice(len(all_combinations), size=num_samples, replace=False)
    return [all_combinations[i] for i in sorted(indices)]


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model_subset(model: torch.nn.Module, loader: DataLoader,
                          device: torch.device, num_classes: int) -> Dict:
    """Evaluate model and compute metrics for class subset."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_labels.append(y.numpy())
            all_preds.append(out.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(out, 1).cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return compute_metrics(y_true, y_pred, y_prob, num_classes)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(output_dir: str, verbose: bool = True, resume: bool = True):
    """Run Experiment 3: Class Scaling Analysis.

    Args:
        output_dir: Output directory for results
        verbose: Print detailed progress
        resume: Skip already completed runs (default True)
    """
    print(f"Device: {DEVICE}")
    print(f"Resume mode: {resume}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Fashion-MNIST...")
    dataset = FashionMNISTClassSubset()

    # Get config
    seeds = CONFIG["seeds"]
    class_counts = CONFIG["exp3"]["class_counts"]
    subset_pcts = CONFIG["exp3"]["subset_percentages"]
    num_class_samples = CONFIG["exp3"]["num_class_samples"]

    total_runs = len(class_counts) * num_class_samples * len(subset_pcts) * len(seeds)
    print(f"Running {len(class_counts)} class counts × {num_class_samples} combinations × "
          f"{len(subset_pcts)} data percentages × {len(seeds)} seeds = {total_runs} runs")

    all_results = []
    skipped_count = 0
    completed_count = 0

    # Generate all class combinations upfront (using a master seed for reproducibility)
    master_seed = 12345
    class_combinations = {}
    for num_classes in class_counts:
        class_combinations[num_classes] = sample_class_combinations(
            num_classes, num_class_samples, total_classes=10, seed=master_seed + num_classes
        )
        print(f"  Classes={num_classes}: {len(class_combinations[num_classes])} combinations")

    # Save class combinations for reference
    combo_file = output_path / "aggregated" / f"class_combinations_n{num_class_samples}.json"
    combo_file.parent.mkdir(parents=True, exist_ok=True)
    save_json({str(k): [list(c) for c in v] for k, v in class_combinations.items()}, str(combo_file))

    # Main experiment loop
    for seed in seeds:
        print(f"\n{'=' * 60}\nSeed: {seed}\n{'=' * 60}")

        seed_dir = output_path / "raw" / f"n{num_class_samples}_seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for num_classes in class_counts:
            combos = class_combinations[num_classes]

            for combo_idx, class_list in enumerate(combos):
                class_list = list(class_list)
                class_names = [CLASS_NAMES[i] for i in class_list]

                print(f"\n  Classes={num_classes}, Combo {combo_idx + 1}/{len(combos)}: {class_list}")
                print(f"    ({', '.join(class_names)})")

                # Get nested indices for this class subset
                set_seed(seed)
                indices_map = dataset.get_nested_indices_for_classes(
                    class_list, [p / 100 for p in subset_pcts], seed
                )

                # Get test loader for this class subset
                test_loader = dataset.get_test_loader_for_classes(class_list, CONFIG["batch_size"])

                for pct in subset_pcts:
                    # Check if already completed (resume mode)
                    filename = f"classes_{num_classes:02d}_combo_{combo_idx:02d}_pct_{pct:03d}.json"
                    result_file = seed_dir / filename

                    if resume and result_file.exists():
                        # Load existing result
                        try:
                            with open(result_file) as f:
                                existing_result = json.load(f)
                            all_results.append(existing_result)
                            skipped_count += 1
                            if verbose:
                                print(f"      {pct}% - SKIPPED (already exists)")
                            continue
                        except (json.JSONDecodeError, KeyError):
                            # File is corrupted, re-run
                            pass

                    indices = indices_map[pct / 100]

                    if verbose:
                        print(f"      {pct}% ({len(indices)} samples)...", end=" ", flush=True)

                    # Create data loaders
                    set_seed(seed)
                    train_loader, val_loader = dataset.create_loaders_for_classes(
                        indices, class_list, CONFIG["batch_size"], CONFIG["val_split"], seed
                    )

                    # Initialize and train model
                    set_seed(seed)
                    model = CNN(num_classes=num_classes, dropout=CONFIG["dropout"]).to(DEVICE)
                    dynamics = train_model(model, train_loader, val_loader, DEVICE, verbose=False)

                    # Evaluate
                    metrics = evaluate_model_subset(model, test_loader, DEVICE, num_classes)

                    # Store results
                    results = {
                        "experiment": "exp3",
                        "condition": {
                            "num_classes": num_classes,
                            "class_list": class_list,
                            "class_names": class_names,
                            "combo_idx": combo_idx,
                            "subset_pct": pct,
                            "num_samples": len(indices),
                            "samples_per_class": len(indices) // num_classes,
                            "train_samples": len(train_loader.dataset),
                            "val_samples": len(val_loader.dataset),
                            "test_samples": len(test_loader.dataset),
                        },
                        "seed": seed,
                        "timestamp": datetime.now().isoformat(),
                        "training_dynamics": {k: v for k, v in dynamics.items() if k != "history"},
                        "training_history": dynamics["history"],
                        **metrics
                    }

                    # Save individual result
                    save_json(results, str(result_file))
                    all_results.append(results)
                    completed_count += 1

                    if verbose:
                        acc = metrics['global_metrics']['accuracy']
                        f1 = metrics['global_metrics']['f1_macro']
                        print(f"Acc: {acc:.4f}, F1: {f1:.4f}")

                    # Memory cleanup after each run
                    del model, train_loader, val_loader, dynamics, metrics
                    cleanup_memory()

                # Cleanup test loader after each class combination
                del test_loader
                cleanup_memory()

    # Save aggregated results
    agg_dir = output_path / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, str(agg_dir / f"all_results_n{num_class_samples}.json"))

    # Create summary CSV
    create_summary_csv(all_results, str(agg_dir / f"summary_n{num_class_samples}.csv"))

    # Create aggregated statistics (mean/std across combinations and seeds)
    create_aggregated_stats(all_results, str(agg_dir / f"aggregated_stats_n{num_class_samples}.csv"))

    # Generate plots
    print("\nGenerating plots...")
    target_accuracies = CONFIG["exp3"].get("target_accuracies", [0.90])
    generate_exp3_plots(str(output_path), str(output_path / "figures"), target_accuracies)

    # Generate optimal points analysis
    print("\nGenerating optimal points analysis...")
    try:
        from find_optimal import process_exp3_curve_fits, save_results_csv, save_results_json
        curve_fits_path = agg_dir / "curve_fits.json"
        if curve_fits_path.exists():
            with open(curve_fits_path) as f:
                curve_fits = json.load(f)
            optimal_results = process_exp3_curve_fits(curve_fits, threshold=0.0001)
            save_results_csv(optimal_results, str(agg_dir / "optimal_points_exp3.csv"))
            save_results_json(optimal_results, str(agg_dir / "optimal_points_exp3.json"))
        else:
            print("  Warning: curve_fits.json not found, skipping optimal points")
    except Exception as e:
        print(f"  Error generating optimal points: {e}")

    print(f"\n{'=' * 60}")
    print(f"Complete! Results: {output_dir}")
    print(f"  Completed: {completed_count} runs")
    print(f"  Skipped (resumed): {skipped_count} runs")
    print(f"  Total: {completed_count + skipped_count} runs")
    print(f"{'=' * 60}")


def create_summary_csv(results: List[Dict], output_path: str) -> None:
    """Create summary CSV with all results."""
    cols = [
        "seed", "num_classes", "combo_idx", "class_list", "subset_pct", "num_samples",
        "samples_per_class", "epochs_trained", "training_time",
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "auroc_macro", "auprc_macro",
        "mcc", "cohens_kappa", "log_loss"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for r in results:
            writer.writerow({
                "seed": r["seed"],
                "num_classes": r["condition"]["num_classes"],
                "combo_idx": r["condition"]["combo_idx"],
                "class_list": str(r["condition"]["class_list"]),
                "subset_pct": r["condition"]["subset_pct"],
                "num_samples": r["condition"]["num_samples"],
                "samples_per_class": r["condition"]["samples_per_class"],
                "epochs_trained": r["training_dynamics"]["epochs_trained"],
                "training_time": round(r["training_dynamics"]["training_time_seconds"], 2),
                "accuracy": round(r["global_metrics"]["accuracy"], 4),
                "balanced_accuracy": round(r["global_metrics"]["balanced_accuracy"], 4),
                "f1_macro": round(r["global_metrics"]["f1_macro"], 4),
                "f1_weighted": round(r["global_metrics"]["f1_weighted"], 4),
                "precision_macro": round(r["global_metrics"]["precision_macro"], 4),
                "recall_macro": round(r["global_metrics"]["recall_macro"], 4),
                "auroc_macro": round(r["global_metrics"]["auroc_macro"], 4),
                "auprc_macro": round(r["global_metrics"]["auprc_macro"], 4),
                "mcc": round(r["global_metrics"]["mcc"], 4),
                "cohens_kappa": round(r["global_metrics"]["cohens_kappa"], 4),
                "log_loss": round(r["global_metrics"]["log_loss"], 4),
            })

    print(f"Summary CSV saved to {output_path}")


def create_aggregated_stats(results: List[Dict], output_path: str) -> None:
    """Create aggregated statistics (mean/std across combinations and seeds)."""
    import pandas as pd

    # Convert to DataFrame
    rows = []
    for r in results:
        rows.append({
            "num_classes": r["condition"]["num_classes"],
            "subset_pct": r["condition"]["subset_pct"],
            "num_samples": r["condition"]["num_samples"],
            "samples_per_class": r["condition"]["samples_per_class"],
            "accuracy": r["global_metrics"]["accuracy"],
            "balanced_accuracy": r["global_metrics"]["balanced_accuracy"],
            "f1_macro": r["global_metrics"]["f1_macro"],
            "auroc_macro": r["global_metrics"]["auroc_macro"],
            "auprc_macro": r["global_metrics"]["auprc_macro"],
            "mcc": r["global_metrics"]["mcc"],
            "precision_macro": r["global_metrics"]["precision_macro"],
            "recall_macro": r["global_metrics"]["recall_macro"],
        })

    df = pd.DataFrame(rows)

    # Aggregate by (num_classes, subset_pct)
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "auroc_macro",
               "auprc_macro", "mcc", "precision_macro", "recall_macro"]

    agg_funcs = {m: ["mean", "std", "min", "max"] for m in metrics}
    agg_funcs["num_samples"] = "first"
    agg_funcs["samples_per_class"] = "first"

    agg_df = df.groupby(["num_classes", "subset_pct"]).agg(agg_funcs)
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]
    agg_df = agg_df.reset_index()

    agg_df.to_csv(output_path, index=False)
    print(f"Aggregated stats saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Class Scaling Analysis")
    parser.add_argument("--output", default="results/exp3", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip completed runs, re-run everything")
    args = parser.parse_args()

    run_experiment(args.output, verbose=not args.quiet, resume=not args.no_resume)
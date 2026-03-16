#!/usr/bin/env python3
"""Experiment 1: Diminishing Returns Analysis.

Usage:
    python exp1_diminishing_returns.py
    python exp1_diminishing_returns.py --output results/exp1 --quiet
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from config import CONFIG, DEVICE
from utils import set_seed, save_json
from dataset import FashionMNISTDataset
from model import CNN
from trainer import train_model
from evaluate import evaluate_model, generate_exp1_plots


def run_experiment(output_dir: str, verbose: bool = True):
    print(f"Device: {DEVICE}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Fashion-MNIST...")
    dataset = FashionMNISTDataset()

    seeds = CONFIG["seeds"]
    subset_pcts = CONFIG["exp1"]["subset_percentages"]      # eg. [5, 10, 15, ... 100]

    print(f"Running {len(subset_pcts)} subsets × {len(seeds)} seeds = {len(subset_pcts) * len(seeds)} runs")

    all_results = []
    test_loader = dataset.get_test_loader(CONFIG["batch_size"])

    for seed in seeds:
        print(f"\n{'=' * 60}\nSeed: {seed}\n{'=' * 60}")
        indices_map = dataset.get_nested_indices([p / 100 for p in subset_pcts], seed)
        seed_dir = output_path / "raw" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for pct in subset_pcts:
            indices = indices_map[pct / 100]
            print(f"\nSubset: {pct}% ({len(indices)} samples)")

            set_seed(seed)
            train_loader, val_loader = dataset.create_loaders(
                indices, CONFIG["batch_size"], CONFIG["val_split"], seed
            )

            set_seed(seed)
            model = CNN(CONFIG["num_classes"], CONFIG["dropout"]).to(DEVICE)
            dynamics = train_model(model, train_loader, val_loader, DEVICE, verbose)
            metrics = evaluate_model(model, test_loader, DEVICE, CONFIG["num_classes"])

            results = {
                "experiment": "exp1",
                "condition": {
                    "subset_pct": pct,
                    "num_samples": len(indices),
                    "train_samples": len(train_loader.dataset),
                    "val_samples": len(val_loader.dataset)
                },
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "training_dynamics": {k: v for k, v in dynamics.items() if k != "history"},
                "training_history": dynamics["history"],
                **metrics
            }

            save_json(results, str(seed_dir / f"pct_{pct:03d}.json"))
            all_results.append(results)
            print(f"  Acc: {metrics['global_metrics']['accuracy']:.4f} | "
                  f"F1: {metrics['global_metrics']['f1_macro']:.4f}")

    # Save aggregated
    agg_dir = output_path / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, str(agg_dir / "all_results.json"))

    # Summary CSV
    cols = ["seed", "subset_pct", "num_samples", "epochs_trained", "training_time",
            "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "auroc_macro", "auprc_macro", "mcc", "cohens_kappa", "log_loss"]
    with open(agg_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_results:
            w.writerow({
                "seed": r["seed"],
                "subset_pct": r["condition"]["subset_pct"],
                "num_samples": r["condition"]["num_samples"],
                "epochs_trained": r["training_dynamics"]["epochs_trained"],
                "training_time": round(r["training_dynamics"]["training_time_seconds"], 2),
                **{k: round(r["global_metrics"][k], 4) for k in cols[5:]}
            })

    # Generate plots
    print("\nGenerating plots...")
    generate_exp1_plots(str(output_path), str(output_path / "figures"))

    print(f"\n{'=' * 60}\nComplete! Results: {output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Diminishing Returns")
    parser.add_argument("--output", default="results/exp1")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_experiment(args.output, not args.quiet)
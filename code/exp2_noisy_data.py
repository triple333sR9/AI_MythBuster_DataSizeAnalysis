#!/usr/bin/env python3
"""Experiment 2: Noisy Data Degradation.

Trains models with clean base data (30k samples) plus varying amounts of noisy data
at different noise rates (5%, 10%, 15%) to demonstrate that more data isn't always better.

Usage:
    python exp2_noisy_data.py
    python exp2_noisy_data.py --output results/exp2 --quiet
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from config import CONFIG, DEVICE
from utils import set_seed, save_json
from dataset import FashionMNISTDataset
from model import CNN
from trainer import train_model
from evaluate import evaluate_model, generate_exp2_plots


def run_experiment(output_dir: str, verbose: bool = True):
    print(f"Device: {DEVICE}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Fashion-MNIST...")
    dataset = FashionMNISTDataset()

    seeds = CONFIG["seeds"]
    clean_samples = CONFIG["exp2"]["clean_samples"]
    noise_rates = CONFIG["exp2"]["noise_rates"]
    noisy_pcts = CONFIG["exp2"]["noisy_percentages"]
    batch_size = CONFIG["batch_size"]
    val_split = CONFIG["val_split"]
    dropout = CONFIG["dropout"]

    total_runs = len(seeds) * len(noise_rates) * len(noisy_pcts)
    print(
        f"Running {len(noise_rates)} noise rates × {len(noisy_pcts)} noisy pcts × {len(seeds)} seeds = {total_runs} runs")
    print(f"Clean samples: {clean_samples} | Noise rates: {[f'{r * 100:.0f}%' for r in noise_rates]}")

    all_results = []
    baseline_metrics = {}
    test_loader = dataset.get_test_loader(batch_size)

    for seed in seeds:
        print(f"\n{'=' * 60}\nSeed: {seed}\n{'=' * 60}")
        seed_dir = output_path / "raw" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Get clean indices (first 50% of data, stratified)
        set_seed(seed)
        all_indices = np.arange(dataset.num_train)
        clean_indices = dataset.get_nested_indices([0.5], seed)[0.5]
        remaining_indices = np.setdiff1d(all_indices, clean_indices)

        for noise_rate in noise_rates:
            noise_pct_int = int(noise_rate * 100)
            print(f"\n--- Noise Rate: {noise_pct_int}% ---")

            for noisy_pct in noisy_pcts:
                # Calculate how many noisy samples to add
                noisy_samples = int(dataset.num_train * noisy_pct / 100)

                if noisy_samples > 0:
                    # Select noisy indices from remaining pool
                    set_seed(seed)
                    if noisy_samples >= len(remaining_indices):
                        noisy_indices = remaining_indices
                    else:
                        noisy_indices = np.random.choice(remaining_indices, size=noisy_samples, replace=False)
                else:
                    noisy_indices = np.array([], dtype=int)

                total_samples = len(clean_indices) + len(noisy_indices)
                print(
                    f"\nNoisy pct: {noisy_pct}% | Clean: {len(clean_indices)} | Noisy: {len(noisy_indices)} | Total: {total_samples}")

                set_seed(seed)
                train_loader, val_loader, num_corrupted = dataset.create_noisy_loaders(
                    clean_indices, noisy_indices, noise_rate, batch_size, val_split, seed
                )

                set_seed(seed)
                model = CNN(CONFIG["num_classes"], dropout).to(DEVICE)
                dynamics = train_model(model, train_loader, val_loader, DEVICE, verbose)
                metrics = evaluate_model(model, test_loader, DEVICE, CONFIG["num_classes"])

                results = {
                    "experiment": "exp2",
                    "condition": {
                        "noise_rate": noise_rate,
                        "noisy_pct": noisy_pct,
                        "clean_samples": len(clean_indices),
                        "noisy_samples": len(noisy_indices),
                        "num_corrupted": num_corrupted,
                        "total_samples": total_samples,
                        "train_samples": len(train_loader.dataset),
                        "val_samples": len(val_loader.dataset)
                    },
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "training_dynamics": {k: v for k, v in dynamics.items() if k != "history"},
                    "training_history": dynamics["history"],
                    **metrics
                }

                filename = f"noise{noise_pct_int:02d}_pct{noisy_pct:02d}.json"
                save_json(results, str(seed_dir / filename))
                all_results.append(results)

                # Store baseline metrics (0% noisy data)
                if noisy_pct == 0 and seed == seeds[0]:
                    baseline_metrics = metrics["global_metrics"].copy()

                print(f"  Acc: {metrics['global_metrics']['accuracy']:.4f} | "
                      f"F1: {metrics['global_metrics']['f1_macro']:.4f} | "
                      f"Corrupted: {num_corrupted}")

    # Save aggregated results
    agg_dir = output_path / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, str(agg_dir / "all_results.json"))
    save_json(baseline_metrics, str(agg_dir / "baseline_metrics.json"))

    # Summary CSV
    cols = ["seed", "noise_rate", "noisy_pct", "clean_samples", "noisy_samples", "num_corrupted",
            "total_samples", "epochs_trained", "training_time",
            "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "auroc_macro", "auprc_macro", "mcc", "cohens_kappa", "log_loss"]
    with open(agg_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_results:
            w.writerow({
                "seed": r["seed"],
                "noise_rate": r["condition"]["noise_rate"],
                "noisy_pct": r["condition"]["noisy_pct"],
                "clean_samples": r["condition"]["clean_samples"],
                "noisy_samples": r["condition"]["noisy_samples"],
                "num_corrupted": r["condition"]["num_corrupted"],
                "total_samples": r["condition"]["total_samples"],
                "epochs_trained": r["training_dynamics"]["epochs_trained"],
                "training_time": round(r["training_dynamics"]["training_time_seconds"], 2),
                **{k: round(r["global_metrics"][k], 4) for k in cols[9:]}
            })

    # Generate plots
    print("\nGenerating plots...")
    generate_exp2_plots(str(output_path), str(output_path / "figures"), baseline_metrics)

    print(f"\n{'=' * 60}\nComplete! Results: {output_dir}\n{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Noisy Data Degradation")
    parser.add_argument("--output", default="results/exp2")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_experiment(args.output, not args.quiet)
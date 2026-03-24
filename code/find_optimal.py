#!/usr/bin/env python3
"""Compute optimal efficiency points from curve fits.

Computes three types of optimal points:
1. Maximum Efficiency Ratio: Where metric/x is maximized
2. Knee/Elbow Point: Where rate of improvement drops significantly
3. Marginal Threshold: Where dy/dx drops below a specified value

Usage:
    python find_optimal.py --all
    python find_optimal.py --exp1 --exp2 --exp3
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize_scalar


# =============================================================================
# Curve Functions
# =============================================================================

def get_curve_func(func_type: str, params: List[float]):
    """Return curve function and derivative."""
    if func_type == "log":
        a, b = params
        return lambda x: a * np.log(x) + b, lambda x: a / x
    elif func_type == "linear":
        a, b = params
        return lambda x: a * x + b, lambda x: a
    elif func_type == "power":
        a, c, b = params
        return lambda x: a * np.power(x, c) + b, lambda x: a * c * np.power(x, c - 1)
    raise ValueError(f"Unknown: {func_type}")


# =============================================================================
# Optimal Point Calculations
# =============================================================================

def find_efficiency(func_type: str, params: List[float], x_min: float, x_max: float) -> Tuple:
    """Find x where metric/x is maximized."""
    f, _ = get_curve_func(func_type, params)
    f_min = f(x_min)

    if func_type == "log":
        res = minimize_scalar(
            lambda x: -(f(x) - f_min) / (x - x_min),
            bounds=(x_min + 1, x_max),
            method='bounded'
        )
        x_opt = res.x
        if x_min <= x_opt <= x_max:
            return x_opt, f(x_opt), "analytical"
        return (x_min, f(x_min), "boundary") if f(x_min) / x_min > f(x_max) / x_max else (x_max, f(x_max), "boundary")
    elif func_type == "linear":
        return (x_min, f(x_min), "boundary") if params[1] >= 0 else (x_max, f(x_max), "boundary")
    else:
        res = minimize_scalar(lambda x: -f(x) / x, bounds=(x_min, x_max), method='bounded')
        return res.x, f(res.x), "numerical"


def find_knee(func_type: str, params: List[float], x_min: float, x_max: float) -> Tuple:
    """Find knee point using geometric distance method."""
    if func_type == "linear":
        return None, None, "no_knee"
    f, _ = get_curve_func(func_type, params)
    x = np.linspace(x_min, x_max, 1000)
    y = f(x)
    x_n, y_n = (x - x_min) / (x_max - x_min), (y - y.min()) / (y.max() - y.min() + 1e-10)
    idx = np.argmax(np.abs(y_n - x_n))
    return x[idx], y[idx], "max_distance"


def find_marginal(func_type: str, params: List[float], x_min: float, x_max: float, thresh: float) -> Tuple:
    """Find x where dy/dx drops below threshold."""
    f, df = get_curve_func(func_type, params)

    if func_type == "log":
        x_t = params[0] / thresh if thresh > 0 else x_max
        if x_t <= x_max:
            return max(x_t, x_min), f(max(x_t, x_min)), f"dy/dx={thresh}"
        return x_max, f(x_max), "not_reached"
    elif func_type == "linear":
        return (x_min, f(x_min), "constant") if abs(params[0]) <= thresh else (None, None, "above_thresh")
    elif func_type == "power":
        a, c, _ = params
        if a * c > 0 and c != 1:
            x_t = np.power(thresh / (a * c), 1 / (c - 1))
            if x_min <= x_t <= x_max:
                return x_t, f(x_t), f"dy/dx={thresh}"
    return None, None, "not_found"


def find_efficiency_exp2(func_type: str, params: List[float], x_min: float, x_max: float,
                         clean: int) -> Tuple:
    """For exp2: maximize metric per noisy/clean ratio."""
    f, _ = get_curve_func(func_type, params)
    x_min = max(x_min, clean + 1)
    f_min = f(x_min)
    if x_min >= x_max:
        return None, None, None, "invalid"
    res = minimize_scalar(lambda x: -(f(x) - f_min) / ((x - clean) / clean), bounds=(x_min + 1, x_max), method='bounded')
    return res.x, f(res.x), (res.x - clean) / clean, "numerical"


# =============================================================================
# Key Parsing
# =============================================================================

def parse_key(key: str, exp: str) -> Dict:
    """Parse curve fit key to extract metadata."""
    if exp == "exp1":
        return {"metric": key}
    elif exp == "exp2":
        if "_baseline" in key:
            return {"metric": key.replace("_baseline", ""), "condition": "baseline", "noise_rate": 0.0}
        if "_noise_" in key:
            m, n = key.split("_noise_")
            return {"metric": m, "condition": f"{n.replace('pct', '%')} noise",
                    "noise_rate": float(n.replace('pct', '')) / 100}
    elif exp == "exp3":
        if "_classes_" in key:
            m, n = key.rsplit("_classes_", 1)
            return {"metric": m, "condition": f"{n} classes", "num_classes": int(n)}
        if "iso_accuracy_" in key:
            return {"metric": "iso_accuracy", "condition": key.replace("iso_accuracy_", "").replace("pct", "% target")}
        if "scaling_law_pct_" in key:
            return {"metric": "scaling_law", "condition": key.replace("scaling_law_pct_", "") + "% data"}
    return {"metric": key, "condition": "unknown"}


# =============================================================================
# Unified Processing
# =============================================================================

def compute_optimal(fit: Dict, thresh: float, exp2_clean: int = None, noise_rate: float = None) -> Dict:
    """Compute all optimal points for a single fit."""
    ft, p, xmin, xmax = fit["func_type"], fit["params"], fit["x_min"], fit["x_max"]

    row = {"func_type": ft, "equation": fit["equation"], "r_squared": round(fit["r_squared"], 4),
           "x_min": xmin, "x_max": xmax, "efficiency_noisy_clean_ratio": None}

    # Efficiency
    if exp2_clean and noise_rate and noise_rate > 0:
        x, y, ratio, note = find_efficiency_exp2(ft, p, xmin, xmax, exp2_clean)
        row["efficiency_noisy_clean_ratio"] = round(ratio, 4) if ratio else None
    else:
        x, y, note = find_efficiency(ft, p, xmin, xmax)
    row.update({"efficiency_x": round(x, 2) if x else None, "efficiency_y": round(y, 4) if y else None,
                "efficiency_note": note})
    if x and y and "efficiency_noisy_clean_ratio" not in row:
        row["efficiency_ratio"] = round(y / x, 8)

    # Knee
    x, y, note = find_knee(ft, p, xmin, xmax)
    row.update({"knee_x": round(x, 2) if x else None, "knee_y": round(y, 4) if y else None, "knee_note": note})

    # Marginal
    x, y, note = find_marginal(ft, p, xmin, xmax, thresh)
    row.update({"marginal_x": round(x, 2) if x else None, "marginal_y": round(y, 4) if y else None,
                "marginal_note": note, "marginal_threshold": thresh})
    return row


def process_curve_fits(curve_fits: Dict, exp: str, thresh: float = 0.0001, clean: int = 30000) -> List[Dict]:
    """Process curve fits for any experiment."""
    results = []
    for key, fit in curve_fits.get("fits", {}).items():
        meta = parse_key(key, exp)
        noise = meta.get("noise_rate") if exp == "exp2" else None
        row = compute_optimal(fit, thresh, clean if exp == "exp2" else None, noise)
        row.update(meta)
        results.append(row)
    return results


# Convenience wrappers
def process_exp1_curve_fits(cf: Dict, thresh: float = 0.0001) -> List[Dict]:
    return process_curve_fits(cf, "exp1", thresh)


def process_exp2_curve_fits(cf: Dict, thresh: float = 0.0001, clean: int = 30000) -> List[Dict]:
    return process_curve_fits(cf, "exp2", thresh, clean)


def process_exp3_curve_fits(cf: Dict, thresh: float = 0.0001) -> List[Dict]:
    return process_curve_fits(cf, "exp3", thresh)


# =============================================================================
# I/O
# =============================================================================

def save_results_csv(results: List[Dict], path: str) -> None:
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {path}")


def save_results_json(results: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def print_summary(results: List[Dict], exp: str) -> None:
    print(f"\n{'=' * 60}\n{exp.upper()} OPTIMAL POINTS\n{'=' * 60}")
    for r in results:
        hdr = r.get("metric", "?") + (f" ({r['condition']})" if r.get("condition") else "")
        print(f"\n{hdr}: {r.get('equation', 'N/A')} (R²={r.get('r_squared', 'N/A')})")
        for t, xk, yk in [("Eff", "efficiency_x", "efficiency_y"), ("Knee", "knee_x", "knee_y"),
                          ("Marg", "marginal_x", "marginal_y")]:
            x, y = r.get(xk), r.get(yk)
            print(f"  {t}: x={x:.1f}, y={y:.4f}" if x else f"  {t}: N/A")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Compute optimal points from curve fits")
    p.add_argument("--exp1", nargs="?", const="results/exp1/aggregated/curve_fits.json")
    p.add_argument("--exp2", nargs="?", const="results/exp2/aggregated/curve_fits.json")
    p.add_argument("--exp3", nargs="?", const="results/exp3/aggregated/curve_fits.json")
    p.add_argument("--exp1-path", default="results/exp1")
    p.add_argument("--exp2-path", default="results/exp2")
    p.add_argument("--exp3-path", default="results/exp3")
    p.add_argument("--all", action="store_true")
    p.add_argument("--threshold", type=float, default=0.0001)
    p.add_argument("--clean-samples", type=int, default=30000)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if not any([args.exp1, args.exp2, args.exp3, args.all]):
        args.all = True

    for exp in ["exp1", "exp2", "exp3"]:
        if getattr(args, exp, None) or args.all:
            path = getattr(args, exp) or f"{getattr(args, f'{exp}_path')}/aggregated/curve_fits.json"
            print(f"\n{'=' * 60}\nProcessing {exp}: {path}\n{'=' * 60}")
            try:
                with open(path) as f:
                    fits = json.load(f)
                results = process_curve_fits(fits, exp, args.threshold, args.clean_samples)
                out = str(Path(path).parent)
                save_results_csv(results, f"{out}/optimal_points_{exp}.csv")
                save_results_json(results, f"{out}/optimal_points_{exp}.json")
                if not args.quiet:
                    print_summary(results, exp)
            except FileNotFoundError:
                print(f"Not found: {path}")
            except Exception as e:
                print(f"Error: {e}")

    print(f"\n{'=' * 60}\nDone!\n{'=' * 60}")


if __name__ == "__main__":
    main()
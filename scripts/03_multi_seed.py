#!/usr/bin/env python3
"""03_multi_seed.py — Multi-seed experiment to confirm stability.

Usage:
    python scripts/03_multi_seed.py --site dkasc_site7
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import make_dataloaders
from sgba.training.protocol import TrainingConfig
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seeds = [42, 123, 7]

    print(f"=== Multi-seed experiment on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=42)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    results_per_seed = {}
    for seed in seeds:
        setup_experiment(seed=seed)
        model = SGBA(
            geometry_mode="elev",
            elev_mean=meta["elev_mean"],
            elev_std=meta["elev_std"],
        )
        results = train_model(model, train_loader, val_loader, test_loader)
        results_per_seed[seed] = {
            "RMSE": results["metrics"]["RMSE"],
            "MAE": results["metrics"]["MAE"],
            "R2": results["metrics"]["R2"],
        }
        print(f"  Seed {seed}: RMSE={results['metrics']['RMSE']:.4f}")

    rmses = [r["RMSE"] for r in results_per_seed.values()]
    print(f"\nMean RMSE: {sum(rmses)/len(rmses):.4f}")
    print(f"Std  RMSE: {(sum((r - sum(rmses)/len(rmses))**2 for r in rmses) / (len(rmses)-1))**0.5:.4f}")

    save_json({
        "site": site,
        "seeds": seeds,
        "results_per_seed": results_per_seed,
        "mean_rmse": sum(rmses)/len(rmses),
        "std_rmse": (sum((r - sum(rmses)/len(rmses))**2 for r in rmses) / (len(rmses)-1))**0.5,
    }, RESULTS_DIR / f"03_multi_seed_{site}.json")


if __name__ == "__main__":
    main()

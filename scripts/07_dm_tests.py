#!/usr/bin/env python3
"""07_dm_tests.py — Diebold-Mariano significance tests.

Usage:
    python scripts/07_dm_tests.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sgba.models import SGBA, PatchTST, DLinear
from sgba.training import train_model, TrainingConfig
from sgba.training.metrics import diebold_mariano_test, block_bootstrap_ci
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seed = int(args.get("seed", 42))
    setup_experiment(seed=seed)
    print(f"=== Diebold-Mariano tests on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    # Train SGBA and baselines
    sgba = SGBA(geometry_mode="elev", elev_mean=meta["elev_mean"], elev_std=meta["elev_std"])
    r_sgba = train_model(sgba, train_loader, val_loader, test_loader, verbose=False)
    err_sgba = r_sgba["errors"]

    baselines = {
        "PatchTST": PatchTST(),
        "DLinear": DLinear(),
    }
    dm_results = {}
    for name, model in baselines.items():
        r = train_model(model, train_loader, val_loader, test_loader, verbose=False)
        err_b = r["errors"]
        dm_stat, p_val = diebold_mariano_test(err_sgba, err_b, max_lag=24)
        mean_diff, ci_lo, ci_hi, std = block_bootstrap_ci(err_sgba, err_b)
        dm_results[name] = {
            "dm_stat": dm_stat, "p_value": p_val,
            "mean_diff_rmse": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n": len(err_sgba),
        }
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  vs {name}: DM={dm_stat:.2f} p={p_val:.4g} {sig}  ΔRMSE={mean_diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    save_json({"site": site, "seed": seed, "dm_tests": dm_results},
              RESULTS_DIR / f"07_dm_tests_{site}_seed{seed}.json")


if __name__ == "__main__":
    main()

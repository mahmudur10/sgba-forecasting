#!/usr/bin/env python3
"""10_alpha_sensitivity.py — α sweep for SGBA-EWMA.

Usage:
    python scripts/10_alpha_sensitivity.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA
from sgba.training import train_model
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


ALPHAS = [0.70, 0.80, 0.90, 0.95]


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seed = int(args.get("seed", 42))
    setup_experiment(seed=seed)
    print(f"=== α sensitivity sweep on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    results = {}
    for alpha in ALPHAS:
        model = SGBA(
            geometry_mode="ewma", alpha=alpha,
            elev_mean=meta["elev_mean"], elev_std=meta["elev_std"],
        )
        r = train_model(model, train_loader, val_loader, test_loader, verbose=False)
        results[alpha] = {
            "RMSE": r["metrics"]["RMSE"],
            "R2": r["metrics"]["R2"],
        }
        print(f"  α={alpha:.2f}: RMSE={r['metrics']['RMSE']:.4f}  R²={r['metrics']['R2']:.4f}")

    save_json({"site": site, "seed": seed, "alphas": results},
              RESULTS_DIR / f"10_alpha_sensitivity_{site}_seed{seed}.json")


if __name__ == "__main__":
    main()

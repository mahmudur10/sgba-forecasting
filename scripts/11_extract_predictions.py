#!/usr/bin/env python3
"""11_extract_predictions.py — Save test predictions + attention for visualization.

Usage:
    python scripts/11_extract_predictions.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sgba.models import SGBA
from sgba.training import train_model
from sgba.data.datasets import make_dataloaders
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seed = int(args.get("seed", 42))
    device = setup_experiment(seed=seed)
    print(f"=== Extracting predictions on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    model = SGBA(
        geometry_mode="elev",
        elev_mean=meta["elev_mean"], elev_std=meta["elev_std"],
    ).to(device)
    results = train_model(model, train_loader, val_loader, test_loader, verbose=False)
    print(f"  Test RMSE: {results['metrics']['RMSE']:.4f}")

    save_json({
        "site": site, "seed": seed,
        "predictions": results["predictions"].tolist(),
        "targets": results["targets"].tolist(),
        "metrics": results["metrics"],
    }, RESULTS_DIR / f"11_predictions_{site}_seed{seed}.json")


if __name__ == "__main__":
    main()

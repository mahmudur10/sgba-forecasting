#!/usr/bin/env python3
"""04_cross_climate.py — Cross-climate validation across 6 sites.

Usage:
    python scripts/04_cross_climate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA, PatchTST, DLinear, LSTM, GRU, MLP, CTNet
from sgba.training import train_model, TrainingConfig
from sgba.data.datasets import make_dataloaders
from sgba.data.solar import SITE_COORDS
from scripts._common import (
    setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


SITES = [
    "dkasc_site7", "dkasc_site10", "dkasc_site2",
    "pvgis_singapore", "pvgis_oslo", "pvgis_madrid",
]


def main():
    setup_experiment(seed=42)
    print("=== Cross-climate validation across 6 sites ===")

    all_results = {}
    for site in SITES:
        print(f"\n--- {site} ({SITE_COORDS[site]['name']}) ---")
        df = make_synthetic_dataset(n_days=200, site=site, seed=42)
        df_train, df_val, df_test = split_chronological(df)
        train_loader, val_loader, test_loader, meta = make_dataloaders(
            df_train, df_val, df_test, site=site,
        )

        site_results = {}
        models = {
            "SGBA-Elev": SGBA(geometry_mode="elev",
                              elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
            "PatchTST": PatchTST(),
            "DLinear": DLinear(),
            "LSTM": LSTM(),
        }
        for name, model in models.items():
            results = train_model(model, train_loader, val_loader, test_loader, verbose=False)
            site_results[name] = results["metrics"]["RMSE"]
            print(f"  {name}: RMSE={results['metrics']['RMSE']:.4f}")
        all_results[site] = site_results

    save_json({"all_sites": all_results},
              RESULTS_DIR / "04_cross_climate.json")
    print(f"\nSaved: {RESULTS_DIR / '04_cross_climate.json'}")


if __name__ == "__main__":
    main()

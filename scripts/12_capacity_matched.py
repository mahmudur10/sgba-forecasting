#!/usr/bin/env python3
"""12_capacity_matched.py — Capacity-matched variants (SGBA shrunk, baselines scaled up).

Usage:
    python scripts/12_capacity_matched.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgba.models import SGBA, PatchTST, LSTM
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
    setup_experiment(seed=seed)
    print(f"=== Capacity-matched comparison on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    # SGBA variants (shrunk d_model)
    sgba_variants = {
        "SGBA-Elev-full (d=96)": SGBA(d_model=96, geometry_mode="elev",
                                      elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
        "SGBA-Elev-d48": SGBA(d_model=48, geometry_mode="elev",
                              elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
        "SGBA-Elev-d32": SGBA(d_model=32, geometry_mode="elev",
                              elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
        "SGBA-Elev-d24": SGBA(d_model=24, geometry_mode="elev",
                              elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
    }
    # Baselines (scaled up)
    baseline_variants = {
        "PatchTST-XL (d=128, 3 layers)": PatchTST(d_model=128, n_blocks=3),
        "LSTM-XL (hidden=128, 3 layers)": LSTM(hidden_dim=128, n_layers=3),
    }

    results = {}
    for name, model in {**sgba_variants, **baseline_variants}.items():
        r = train_model(model, train_loader, val_loader, test_loader, verbose=False)
        results[name] = {
            "params": model.count_parameters(),
            "val_rmse_norm": float(min(r["history"]["val_loss"]) ** 0.5),
            "test_rmse": r["metrics"]["RMSE"],
        }
        print(f"  {name}: params={model.count_parameters():,}  test_rmse={r['metrics']['RMSE']:.4f}")

    save_json({"site": site, "seed": seed, "variants": results},
              RESULTS_DIR / f"12_capacity_matched_{site}_seed{seed}.json")


if __name__ == "__main__":
    main()

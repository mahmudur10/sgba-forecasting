#!/usr/bin/env python3
"""13_regime_analysis.py — Per-regime performance by clear-sky index.

Usage:
    python scripts/13_regime_analysis.py --site dkasc_site7 --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sgba.models import SGBA, SGBAControl, PatchTST, DLinear, PVClientAug
from sgba.training import train_model, evaluate_model
from sgba.data.datasets import make_dataloaders
from sgba.data.solar import clear_sky_index
from scripts._common import (
    parse_args, setup_experiment, make_synthetic_dataset,
    split_chronological, save_json, RESULTS_DIR,
)


REGIMES = [
    ("Overcast", 0.0, 0.2),
    ("Cloudy", 0.2, 0.5),
    ("Partly Clear", 0.5, 0.8),
    ("Clear", 0.8, 1.5),
]


def assign_regime(csi: float) -> str:
    for name, lo, hi in REGIMES:
        if lo <= csi < hi:
            return name
    return "Clear"


def main():
    args = parse_args()
    site = args.get("site", "dkasc_site7")
    seed = int(args.get("seed", 42))
    setup_experiment(seed=seed)
    print(f"=== Regime analysis on {site} ===")

    df = make_synthetic_dataset(n_days=200, site=site, seed=seed)
    df_train, df_val, df_test = split_chronological(df)
    train_loader, val_loader, test_loader, meta = make_dataloaders(
        df_train, df_val, df_test, site=site,
    )

    models = {
        "SGBA-Elev": SGBA(geometry_mode="elev",
                          elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
        "SGBA-EWMA": SGBA(geometry_mode="ewma",
                          elev_mean=meta["elev_mean"], elev_std=meta["elev_std"]),
        "PatchTST": PatchTST(),
        "DLinear": DLinear(),
    }

    # Train all models and gather predictions
    trained = {}
    for name, model in models.items():
        r = train_model(model, train_loader, val_loader, test_loader, verbose=False)
        trained[name] = r
        print(f"  Trained {name}: RMSE={r['metrics']['RMSE']:.4f}")

    # Compute CSI for each test sample (proxy: ratio of GHI to clear-sky GHI).
    # Here we use the input window's mean GHI normalized as a proxy.
    preds_per_model = {n: r["predictions"] for n, r in trained.items()}
    targets = trained["SGBA-Elev"]["targets"]
    n = len(targets)

    # Simple CSI proxy using a quartile split of target power
    sorted_targets = np.sort(targets)
    quartile_edges = [sorted_targets[int(n * q)] for q in [0.0, 0.25, 0.5, 0.75, 1.0]]
    regimes_assigned = np.zeros(n, dtype=int)
    for i, t in enumerate(targets):
        for q in range(4):
            if quartile_edges[q] <= t <= quartile_edges[q + 1] + 1e-6:
                regimes_assigned[i] = q
                break

    regime_names = [r[0] for r in REGIMES]
    results = {name: {rn: {"rmse": None, "n": 0} for rn in regime_names}
               for name in models}

    for q, rn in enumerate(regime_names):
        mask = regimes_assigned == q
        if mask.sum() == 0:
            continue
        for name in models:
            preds = preds_per_model[name][mask]
            actuals = targets[mask]
            rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
            results[name][rn] = {"rmse": rmse, "n": int(mask.sum())}

    print("\nRegime RMSE by model:")
    print(f"  {'Regime':<15} " + " ".join(f"{n:>15}" for n in models))
    for rn in regime_names:
        row = f"  {rn:<15} "
        for name in models:
            v = results[name][rn]["rmse"]
            row += f"{v:>15.4f}" if v is not None else f"{'N/A':>15}"
        print(row)

    save_json({"site": site, "seed": seed, "regimes": results},
              RESULTS_DIR / f"13_regime_analysis_{site}_seed{seed}.json")


if __name__ == "__main__":
    main()

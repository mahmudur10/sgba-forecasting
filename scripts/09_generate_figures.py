#!/usr/bin/env python3
"""09_generate_figures.py — Generate all publication figures.

Reads result JSONs from results/ and produces PDF + PNG figures in figures/.

Usage:
    python scripts/09_generate_figures.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts._common import RESULTS_DIR, setup_experiment

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D1D5DB", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.labelsize": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlepad": 15, "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.5,
})

C = {
    "SGBA-Elev": "#0077BB", "SGBA-EWMA": "#33BBEE",
    "PatchTST": "#009988", "DLinear": "#888888",
    "Persistence": "#795548", "LSTM": "#5D6D7E",
    "GRU": "#117A65", "MLP": "#ABB2B9", "XGBoost": "#8E44AD",
    "CT-Net": "#9B59B6", "PV-Client-Aug": "#CC3311",
    "iTransformer-LSTM-KAN-Aug": "#EE3377",
    "CSIPersistence": "#A1887F", "SGBA-ShuffledElev": "#EE7733",
}


def save_fig(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"  saved {name}.png ({path.stat().st_size // 1024} KB)")


def load_metric(site, model, seed, metric="R2"):
    """Load a metric from result JSON."""
    path = RESULTS_DIR / f"{site}_{model}_seed{seed}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        r = json.load(f)
    tm = r.get("test_metrics", {})
    for key in [metric, metric.upper(), metric.lower()]:
        if key in tm:
            return tm[key]
        if key in r:
            return r[key]
    return None


def fig1_cross_climate_heatmap():
    """Cross-climate R² heatmap."""
    ALL_MODELS = [
        "Persistence", "CSIPersistence", "DLinear", "CT-Net", "PatchTST",
        "LSTM", "GRU", "MLP", "XGBoost",
        "PV-Client-Aug", "iTransformer-LSTM-KAN-Aug",
        "SGBA-EWMA", "SGBA-Elev",
    ]
    ALL_SITES = ["site7", "site10", "site2", "singapore", "oslo", "madrid"]
    SITE_LABELS = {
        "site7": "DKASC S7\n(Desert)", "site10": "DKASC S10\n(Desert)",
        "site2": "DKASC S2\n(Desert)", "singapore": "Singapore\n(Tropical)",
        "oslo": "Oslo\n(Cold)", "madrid": "Madrid\n(Med.)",
    }

    r2_matrix = np.full((len(ALL_MODELS), len(ALL_SITES)), np.nan)
    for i, model in enumerate(ALL_MODELS):
        for j, site in enumerate(ALL_SITES):
            r2s = [load_metric(site, model, s, "R2") for s in [42, 123, 7]]
            r2s = [r for r in r2s if r is not None]
            if r2s:
                r2_matrix[i, j] = np.mean(r2s)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = LinearSegmentedColormap.from_list("rg", ["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"], N=256)
    im = ax.imshow(r2_matrix, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    for i in range(len(ALL_MODELS)):
        for j in range(len(ALL_SITES)):
            val = r2_matrix[i, j]
            if not np.isnan(val):
                tc = "white" if val < 0.4 or val > 0.85 else "black"
                wt = "bold" if ALL_MODELS[i] == "SGBA-Elev" else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=tc, fontweight=wt)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#CCC")
    ax.set_xticks(range(len(ALL_SITES)))
    ax.set_xticklabels([SITE_LABELS[s] for s in ALL_SITES], fontsize=9)
    ax.set_yticks(range(len(ALL_MODELS)))
    ax.set_yticklabels(ALL_MODELS, fontsize=10)
    ax.set_title(f"Cross-Climate R² Comparison\n({len(ALL_MODELS)} Models × 6 Sites)", fontsize=14, pad=15)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("R²", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    plt.tight_layout(pad=2)
    save_fig(fig, "fig1_cross_climate_heatmap")


def fig2_ablation():
    """Ablation: SGBA-Elev vs SGBA-ShuffledElev."""
    sites = ["site7", "site10", "site2"]
    sgba_vals, shuf_vals = [], []
    for site in sites:
        s = [load_metric(site, "SGBA-Elev", sd, "RMSE") for sd in [42, 123, 7]]
        h = [load_metric(site, "SGBA-ShuffledElev", sd, "RMSE") for sd in [42, 123, 7]]
        s = [v for v in s if v is not None]
        h = [v for v in h if v is not None]
        if s:
            sgba_vals.append((np.mean(s), np.std(s, ddof=1) if len(s) > 1 else 0))
        if h:
            shuf_vals.append((np.mean(h), np.std(h, ddof=1) if len(h) > 1 else 0))

    if not sgba_vals:
        print("  Skipping fig2 — no ablation data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(sgba_vals))
    w = 0.3
    bars1 = ax.bar(x - w/2, [v[0] for v in sgba_vals], w, yerr=[v[1] for v in sgba_vals],
                   capsize=8, label="SGBA-Elev (headline)", color=C["SGBA-Elev"], alpha=0.85,
                   edgecolor="#003366", linewidth=1.5)
    if shuf_vals:
        bars2 = ax.bar(x + w/2, [v[0] for v in shuf_vals], w, yerr=[v[1] for v in shuf_vals],
                       capsize=8, label="SGBA-ShuffledElev (null control)", color=C["SGBA-ShuffledElev"],
                       alpha=0.85, edgecolor="#003366", linewidth=1.5)
    ax.set_ylabel("Test RMSE (kW)", fontsize=12)
    ax.set_title("Ablation: Elevation Signal vs Null Control (3 Seeds)", fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Site {s.replace('site','')}" for s in sites], fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=2)
    save_fig(fig, "fig2_ablation_shuffled")


def fig3_multihorizon():
    """Multi-horizon: SGBA-Elev vs PatchTST."""
    horizons = [1, 6, 12, 24]
    sgba_mh, sgba_std, ptst_mh, ptst_std = [], [], [], []
    for h in horizons:
        s, p = [], []
        for site in ["site7", "site10", "site2"]:
            sv = load_metric(f"mh_{site}_h{h}", "SGBA-Elev", 42, "RMSE")
            pv = load_metric(f"mh_{site}_h{h}", "PatchTST", 42, "RMSE")
            if sv is not None:
                s.append(sv)
            if pv is not None:
                p.append(pv)
        sgba_mh.append(np.mean(s) if s else 0)
        sgba_std.append(np.std(s, ddof=1) if len(s) > 1 else 0)
        ptst_mh.append(np.mean(p) if p else 0)
        ptst_std.append(np.std(p, ddof=1) if len(p) > 1 else 0)

    if max(sgba_mh) == 0 and max(ptst_mh) == 0:
        print("  Skipping fig3 — no multi-horizon data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(horizons, sgba_mh, yerr=sgba_std, marker="o", markersize=12, linewidth=3,
                capsize=8, color=C["SGBA-Elev"], label="SGBA-Elev", zorder=5)
    ax.errorbar(horizons, ptst_mh, yerr=ptst_std, marker="s", markersize=12, linewidth=3,
                capsize=8, color=C["PatchTST"], label="PatchTST")
    ax.fill_between(horizons, [m-s for m,s in zip(sgba_mh,sgba_std)],
                    [m+s for m,s in zip(sgba_mh,sgba_std)], alpha=0.1, color=C["SGBA-Elev"])
    ax.fill_between(horizons, [m-s for m,s in zip(ptst_mh,ptst_std)],
                    [m+s for m,s in zip(ptst_mh,ptst_std)], alpha=0.1, color=C["PatchTST"])
    ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax.set_ylabel("RMSE (kW, mean across 3 sites)", fontsize=12)
    ax.set_title("Multi-Horizon Forecasting: SGBA-Elev vs PatchTST", fontsize=14, pad=15)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}h ahead" for h in horizons], fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=2)
    save_fig(fig, "fig3_multihorizon")


def fig10_statistical_significance():
    """Statistical significance forest plot."""
    stats_path = RESULTS_DIR / "statistical_tests_summary.json"
    if not stats_path.exists():
        print("  Skipping fig10 — no statistical tests data")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 14))
    y_pos = 0
    y_labels, y_positions, bar_vals, colors_list = [], [], [], []

    site_order = ["singapore", "oslo", "madrid", "site7", "site10", "site2"]
    SITE_LABELS_SHORT = {"site7": "S7", "site10": "S10", "site2": "S2",
                         "singapore": "SG", "oslo": "Oslo", "madrid": "Mad"}
    baseline_order = ["Persistence", "CSIPersistence", "DLinear", "PatchTST",
                      "PV-Client-Aug", "iTransformer-LSTM-KAN-Aug"]

    for site in site_order:
        sd = stats.get("sites", {}).get(site, stats.get(site, {}))
        comps = sd.get("comparisons", sd)
        sl = SITE_LABELS_SHORT.get(site, site)
        for bl in baseline_order:
            if bl in comps:
                m = comps[bl]
                imp = m.get("relative_improvement_pct", m.get("improvement_pct", 0))
                p = m.get("p_value", 1)
                y_positions.append(y_pos)
                y_labels.append(f"{sl} vs {bl[:12]}")
                bar_vals.append(imp)
                if p < 0.001: colors_list.append("#0077BB")
                elif p < 0.01: colors_list.append("#33BBEE")
                elif p < 0.05: colors_list.append("#88CCDD")
                else: colors_list.append("#CCCCCC")
                y_pos += 1

    if not y_positions:
        print("  Skipping fig10 — no comparison data found")
        plt.close(fig)
        return

    ax.barh(y_positions, bar_vals, color=colors_list, height=0.7, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Relative Improvement (%)", fontsize=12)
    ax.set_title("Statistical Significance: SGBA-Elev vs All Baselines\n(DM test, Newey-West HAC)", fontsize=13, pad=15)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#0077BB", label="p < 0.001 (***)"),
        Patch(facecolor="#33BBEE", label="p < 0.01 (**)"),
        Patch(facecolor="#88CCDD", label="p < 0.05 (*)"),
        Patch(facecolor="#CCCCCC", label="not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout(pad=2)
    save_fig(fig, "fig10_statistical_significance")


def main():
    setup_experiment(seed=42)
    print("=== Generating publication figures ===")

    figures = [
        ("fig1_cross_climate_heatmap", fig1_cross_climate_heatmap),
        ("fig2_ablation", fig2_ablation),
        ("fig3_multihorizon", fig3_multihorizon),
        ("fig10_statistical_significance", fig10_statistical_significance),
    ]
    for name, fn in figures:
        try:
            fn()
        except Exception as e:
            print(f"  WARNING {name}: {e}")

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

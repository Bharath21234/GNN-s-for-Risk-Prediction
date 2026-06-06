"""
Step 5 — Evaluate and compare all models.

Computes accuracy metrics (RMSE, MAE, MAPE, QLIKE) and statistical backtests
(Kupiec POF, Christoffersen CC) for GNN, GARCH(1,1), and GJR-GARCH.
Produces publication-style comparison tables and plots.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.metrics import compute_accuracy_metrics, backtest_model

os.makedirs(config.RESULTS_DIR + "plots/", exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")
COLORS = {"GNN (GraphSAGE)": "#2196F3", "GARCH(1,1)": "#F44336", "GJR-GARCH": "#FF9800"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _align(pred_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align prediction and target DataFrames on dates and tickers."""
    common_dates   = pred_df.index.intersection(target_df.index)
    common_tickers = pred_df.columns.intersection(target_df.columns)
    p = pred_df.loc[common_dates, common_tickers].values.flatten()
    t = target_df.loc[common_dates, common_tickers].values.flatten()
    return t, p


def _pct_improvement(base_rmse: float, model_rmse: float) -> float:
    return (base_rmse - model_rmse) / base_rmse * 100


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading predictions and targets...")

    gnn_pred    = pd.read_parquet(config.RESULTS_DIR + "gnn_test_predictions.parquet")
    garch_pred  = pd.read_parquet(config.RESULTS_DIR + "garch_test_predictions.parquet")
    gjr_pred    = pd.read_parquet(config.RESULTS_DIR + "gjr_test_predictions.parquet")
    targets     = pd.read_parquet(config.RESULTS_DIR + "targets_test.parquet")
    returns_raw = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")

    # Align all predictions to common dates/tickers
    gnn_pred   = gnn_pred.reindex(index=targets.index,   columns=targets.columns)
    garch_pred = garch_pred.reindex(index=targets.index, columns=targets.columns)
    gjr_pred   = gjr_pred.reindex(index=targets.index,   columns=targets.columns)

    # ── Accuracy metrics ───────────────────────────────────────────────────────
    print("\n── Accuracy Metrics (Test Set) ──────────────────────────────────────")

    yt_gnn,  yp_gnn  = _align(gnn_pred,   targets)
    yt_g11,  yp_g11  = _align(garch_pred, targets)
    yt_gjr,  yp_gjr  = _align(gjr_pred,   targets)

    metrics = [
        compute_accuracy_metrics(yt_gnn, yp_gnn, "GNN (GraphSAGE)"),
        compute_accuracy_metrics(yt_g11, yp_g11, "GARCH(1,1)"),
        compute_accuracy_metrics(yt_gjr, yp_gjr, "GJR-GARCH"),
    ]
    metrics_df = pd.DataFrame(metrics).set_index("model")

    garch_rmse = metrics_df.loc["GARCH(1,1)", "rmse"]
    gjr_rmse   = metrics_df.loc["GJR-GARCH",  "rmse"]
    gnn_rmse   = metrics_df.loc["GNN (GraphSAGE)", "rmse"]

    metrics_df["vs GARCH(1,1) [RMSE Δ%]"] = metrics_df["rmse"].apply(
        lambda r: f"{_pct_improvement(garch_rmse, r):+.1f}%"
    )
    metrics_df["vs GJR-GARCH [RMSE Δ%]"] = metrics_df["rmse"].apply(
        lambda r: f"{_pct_improvement(gjr_rmse, r):+.1f}%"
    )

    print(metrics_df.to_string(float_format=lambda x: f"{x:.5f}"))
    metrics_df.to_csv(config.RESULTS_DIR + "accuracy_metrics.csv")

    print(f"\n  GNN improves over GARCH(1,1) by  "
          f"{_pct_improvement(garch_rmse, gnn_rmse):+.1f}% RMSE")
    print(f"  GNN improves over GJR-GARCH  by  "
          f"{_pct_improvement(gjr_rmse, gnn_rmse):+.1f}% RMSE")

    # ── Backtesting ────────────────────────────────────────────────────────────
    print("\n── Backtesting (Kupiec POF + Christoffersen CC) ─────────────────────")

    bt_gnn   = backtest_model(gnn_pred,   returns_raw, horizon=config.HORIZON, alpha=1-config.CVAR_ALPHA)
    bt_garch = backtest_model(garch_pred, returns_raw, horizon=config.HORIZON, alpha=1-config.CVAR_ALPHA)
    bt_gjr   = backtest_model(gjr_pred,   returns_raw, horizon=config.HORIZON, alpha=1-config.CVAR_ALPHA)

    print("\n  Kupiec POF test (H0: violation rate = 5%; pass = model correct):")
    for name, bt in [("GNN",         bt_gnn),
                     ("GARCH(1,1)",  bt_garch),
                     ("GJR-GARCH",   bt_gjr)]:
        s = bt["summary"]
        print(f"    {name:20s}  viol_rate={s['mean_violation_rate']*100:.2f}%  "
              f"kupiec_pass={s['kupiec_pass_pct']:.1f}%  "
              f"cc_pass={s['cc_pass_pct']:.1f}%")

    bt_summary = {
        "GNN (GraphSAGE)": bt_gnn["summary"],
        "GARCH(1,1)":      bt_garch["summary"],
        "GJR-GARCH":       bt_gjr["summary"],
    }
    with open(config.RESULTS_DIR + "backtest_summary.json", "w") as f:
        json.dump(bt_summary, f, indent=2)

    # ── Sector-level RMSE breakdown ────────────────────────────────────────────
    print("\n── Sector-level RMSE (GNN vs best GARCH) ────────────────────────────")
    sector_map = {t: config.STOCKS.get(t, "Unknown") for t in targets.columns}
    rows = []
    for sec in sorted(set(sector_map.values())):
        tickers_sec = [t for t, s in sector_map.items() if s == sec and t in targets.columns]
        if not tickers_sec:
            continue
        yt = targets[tickers_sec].values.flatten()
        yp_gnn_s   = gnn_pred[tickers_sec].values.flatten()
        yp_garch_s = garch_pred[tickers_sec].values.flatten()
        yp_gjr_s   = gjr_pred[tickers_sec].values.flatten()
        mask = np.isfinite(yt) & np.isfinite(yp_gnn_s) & np.isfinite(yp_garch_s)
        rows.append({
            "sector":          sec,
            "n_stocks":        len(tickers_sec),
            "GNN RMSE":        float(np.sqrt(np.mean((yt[mask] - yp_gnn_s[mask])**2))),
            "GARCH RMSE":      float(np.sqrt(np.mean((yt[mask] - yp_garch_s[mask])**2))),
            "GJR RMSE":        float(np.sqrt(np.mean((yt[mask] - yp_gjr_s[mask])**2))),
        })
    sector_df = pd.DataFrame(rows).set_index("sector")
    sector_df["GNN vs GARCH Δ%"] = sector_df.apply(
        lambda r: f"{_pct_improvement(r['GARCH RMSE'], r['GNN RMSE']):+.1f}%", axis=1
    )
    print(sector_df.to_string(float_format=lambda x: f"{x:.5f}"))
    sector_df.to_csv(config.RESULTS_DIR + "sector_rmse.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_rmse_bar(metrics_df)
    _plot_cvar_timeseries(gnn_pred, garch_pred, gjr_pred, targets)
    _plot_sector_heatmap(sector_df)
    _plot_violation_rates(bt_gnn, bt_garch, bt_gjr, targets.columns.tolist())
    _plot_scatter(yt_gnn, yp_gnn, yt_g11, yp_g11, yt_gjr, yp_gjr)

    print(f"\nAll results saved to {config.RESULTS_DIR}")
    print("Step 5 complete.\n")


# ── Plot functions ─────────────────────────────────────────────────────────────

def _plot_rmse_bar(metrics_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metric_cols = ["rmse", "mae", "mape", "qlike"]
    metric_labels = ["RMSE", "MAE", "MAPE (%)", "QLIKE"]

    for ax, col, label in zip(axes, metric_cols, metric_labels):
        vals   = metrics_df[col].values
        models = metrics_df.index.tolist()
        colors = [COLORS.get(m, "gray") for m in models]
        bars   = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(label)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle("Model Comparison — Test Set Accuracy Metrics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "plots/accuracy_comparison.png", dpi=150)
    plt.close(fig)


def _plot_cvar_timeseries(gnn, garch, gjr, targets):
    # Average CVaR across all stocks per day
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(targets.mean(axis=1), label="Target CVaR",   color="black",                 linewidth=1.2, alpha=0.8)
    ax.plot(gnn.mean(axis=1),     label="GNN (GraphSAGE)", color=COLORS["GNN (GraphSAGE)"], linewidth=1.2)
    ax.plot(garch.mean(axis=1),   label="GARCH(1,1)",    color=COLORS["GARCH(1,1)"],    linewidth=1.0, linestyle="--")
    ax.plot(gjr.mean(axis=1),     label="GJR-GARCH",     color=COLORS["GJR-GARCH"],     linewidth=1.0, linestyle=":")

    ax.set_title("Average 5-day CVaR (95%) — Test Period", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("CVaR (5-day, 95%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "plots/cvar_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_sector_heatmap(sector_df: pd.DataFrame):
    plot_data = sector_df[["GNN RMSE", "GARCH RMSE", "GJR RMSE"]].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        plot_data,
        annot=True, fmt=".4f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, cbar_kws={"label": "RMSE"}
    )
    ax.set_title("Sector-level RMSE Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "plots/sector_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_violation_rates(bt_gnn, bt_garch, bt_gjr, tickers):
    models = {"GNN (GraphSAGE)": bt_gnn, "GARCH(1,1)": bt_garch, "GJR-GARCH": bt_gjr}
    fig, ax = plt.subplots(figsize=(12, 4))

    for name, bt in models.items():
        rates = [bt["kupiec"][t]["violation_rate"] * 100
                 for t in tickers if t in bt["kupiec"]]
        ax.plot(sorted(rates), label=name, linewidth=1.5, color=COLORS.get(name))

    ax.axhline(5.0, color="black", linewidth=1.5, linestyle="--", label="Target 5%")
    ax.set_title("Kupiec Violation Rates by Stock (sorted)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Stock rank")
    ax.set_ylabel("Violation rate (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "plots/violation_rates.png", dpi=150)
    plt.close(fig)


def _plot_scatter(yt_gnn, yp_gnn, yt_g11, yp_g11, yt_gjr, yp_gjr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    datasets = [
        ("GNN (GraphSAGE)", yt_gnn, yp_gnn, COLORS["GNN (GraphSAGE)"]),
        ("GARCH(1,1)",      yt_g11, yp_g11, COLORS["GARCH(1,1)"]),
        ("GJR-GARCH",       yt_gjr, yp_gjr, COLORS["GJR-GARCH"]),
    ]

    for ax, (name, yt, yp, color) in zip(axes, datasets):
        mask = np.isfinite(yt) & np.isfinite(yp)
        # Subsample for speed
        idx = np.random.choice(mask.sum(), size=min(5000, mask.sum()), replace=False)
        ax.scatter(yt[mask][idx], yp[mask][idx], alpha=0.2, s=3, color=color)
        lim = max(np.nanpercentile(yt, 99), np.nanpercentile(yp, 99))
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        rmse_val = float(np.sqrt(np.mean((yt[mask] - yp[mask])**2)))
        ax.set_title(f"{name}\nRMSE={rmse_val:.4f}", fontsize=10)
        ax.set_xlabel("Target CVaR")
        ax.set_ylabel("Predicted CVaR")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)

    fig.suptitle("Predicted vs Target CVaR — Test Set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "plots/scatter_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

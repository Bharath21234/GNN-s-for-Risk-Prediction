"""
Step 4 — Fit GARCH baselines on the test window.

Baseline 1 — GARCH(1,1)  [our own specification]
Baseline 2 — GJR-GARCH(1,1)  [Glosten, Jagannathan & Runkle (1993)]

Both use an expanding-window walk-forward scheme: the model is re-fit every
GARCH_REFIT_FREQ trading days on all available history, producing 1-step-ahead
conditional variance forecasts between refits.

Runtime: ~10–25 minutes (100 stocks × expanding fits).
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.garch import run_garch, run_gjr_garch

os.makedirs(config.RESULTS_DIR, exist_ok=True)


def main():
    # ── Load returns ───────────────────────────────────────────────────────────
    print("Loading returns...")
    returns = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")
    tickers = pd.read_parquet(config.DATA_DIR + "processed/tickers.parquet")["ticker"].tolist()
    returns = returns[[t for t in tickers if t in returns.columns]]

    # Determine test start date (same as GNN evaluation window)
    dates      = pd.read_parquet(config.DATA_DIR + "processed/dates.parquet")["date"]
    test_idx   = np.load(config.DATA_DIR + "processed/test_indices.npy")
    test_start = pd.Timestamp(dates.iloc[test_idx[0]])

    # We also need val predictions for apple-to-apple comparison with GNN val set
    val_idx    = np.load(config.DATA_DIR + "processed/val_indices.npy")
    val_start  = pd.Timestamp(dates.iloc[val_idx[0]])

    print(f"  Val  start: {val_start.date()}")
    print(f"  Test start: {test_start.date()}")
    print(f"  Stocks    : {len(returns.columns)}\n")

    # ── GARCH(1,1) ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Running GARCH(1,1) — our baseline...")
    print("=" * 60)
    garch_preds_val  = run_garch(returns, str(val_start.date()),  horizon=config.HORIZON)
    garch_preds_test = run_garch(returns, str(test_start.date()), horizon=config.HORIZON)

    garch_preds_val.to_parquet(config.RESULTS_DIR  + "garch_val_predictions.parquet")
    garch_preds_test.to_parquet(config.RESULTS_DIR + "garch_test_predictions.parquet")
    print(f"  Saved GARCH predictions → {config.RESULTS_DIR}garch_*_predictions.parquet\n")

    # ── GJR-GARCH(1,1) ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("Running GJR-GARCH(1,1) — Glosten, Jagannathan & Runkle (1993)...")
    print("=" * 60)
    gjr_preds_val  = run_gjr_garch(returns, str(val_start.date()),  horizon=config.HORIZON)
    gjr_preds_test = run_gjr_garch(returns, str(test_start.date()), horizon=config.HORIZON)

    gjr_preds_val.to_parquet(config.RESULTS_DIR  + "gjr_val_predictions.parquet")
    gjr_preds_test.to_parquet(config.RESULTS_DIR + "gjr_test_predictions.parquet")
    print(f"  Saved GJR-GARCH predictions → {config.RESULTS_DIR}gjr_*_predictions.parquet\n")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("── Sanity check (mean predicted CVaR on test set) ───────────────────")
    print(f"  GARCH    mean CVaR: {garch_preds_test.mean().mean():.4f}")
    print(f"  GJR-GARCH mean CVaR: {gjr_preds_test.mean().mean():.4f}")

    targets_test = pd.read_parquet(config.RESULTS_DIR + "targets_test.parquet")
    print(f"  Target   mean CVaR: {targets_test.mean().mean():.4f}")

    print("\nStep 4 complete.\n")


if __name__ == "__main__":
    main()

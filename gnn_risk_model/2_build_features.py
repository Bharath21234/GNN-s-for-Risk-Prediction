"""
Step 2 — Feature engineering, graph pre-computation, and target construction.

Reads raw price/return data, computes node features and CVaR targets for every
valid time step, pre-computes graph topology snapshots, and saves everything to
data/processed/.

Runtime: ~3–8 minutes.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.data   import build_dataset, normalize_features
from src.graph  import precompute_edge_indices

os.makedirs(config.DATA_DIR + "processed", exist_ok=True)


def main():
    # ── Load raw data ──────────────────────────────────────────────────────────
    print("Loading raw data...")
    returns = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")
    spy_ret = pd.read_parquet(config.DATA_DIR + "raw/spy_returns.parquet")["SPY"]
    vix     = pd.read_parquet(config.DATA_DIR + "raw/vix.parquet")["VIX"]

    # Align SPY and VIX to return dates
    spy_ret = spy_ret.reindex(returns.index, method="ffill").fillna(0.0)
    vix     = vix.reindex(returns.index, method="ffill").fillna(20.0)

    # Keep only tickers present in both config and downloaded data
    valid_tickers = [t for t in config.STOCKS if t in returns.columns]
    returns       = returns[valid_tickers]
    sectors       = {t: config.STOCKS[t] for t in valid_tickers}
    all_sectors   = config.SECTORS

    print(f"  Stocks: {len(valid_tickers)}")
    print(f"  Dates : {returns.index[0].date()} → {returns.index[-1].date()}")

    # ── Build features and targets ─────────────────────────────────────────────
    print("\nBuilding features, CVaR targets and forward returns...")
    features, targets, fwd_returns, dates = build_dataset(
        returns     = returns,
        spy_returns = spy_ret,
        vix         = vix,
        sectors     = sectors,
        all_sectors = all_sectors,
        lookback    = config.LOOKBACK_WINDOW,
        alpha       = config.CVAR_ALPHA,
        horizon     = config.HORIZON,
        verbose     = True,
    )
    print(f"  Feature shape : {features.shape}   (T, N, F)")
    print(f"  Target shape  : {targets.shape}    (T, N)  [rolling CVaR, reconstruction]")
    print(f"  Fwd-return    : {fwd_returns.shape}    (T, N)  [realised {config.HORIZON}-day, primary]")

    # ── Train / val / test split ───────────────────────────────────────────────
    train_mask = dates <= config.TRAIN_END
    val_mask   = (dates > config.TRAIN_END) & (dates <= config.VAL_END)
    test_mask  = dates > config.VAL_END

    train_indices = np.where(train_mask)[0]
    val_indices   = np.where(val_mask)[0]
    test_indices  = np.where(test_mask)[0]

    # Drop the last HORIZON-1 days of each split so no realised forward window
    # crosses a split boundary or runs off the end of the series (no leakage,
    # no NaN forward labels).  ~4 days per split — negligible.
    h = config.HORIZON
    def _trim(idx):
        return idx[:-(h - 1)] if h > 1 and len(idx) > (h - 1) else idx
    train_indices = _trim(train_indices)
    val_indices   = _trim(val_indices)
    test_indices  = _trim(test_indices)

    print(f"\n  Train: {len(train_indices)} days  "
          f"({dates[train_indices][0].date()} → {dates[train_indices][-1].date()})")
    print(f"  Val  : {len(val_indices)} days  "
          f"({dates[val_indices][0].date()} → {dates[val_indices][-1].date()})")
    print(f"  Test : {len(test_indices)} days  "
          f"({dates[test_indices][0].date()} → {dates[test_indices][-1].date()})")

    # ── Normalize features (fit on train only) ─────────────────────────────────
    print("\nNormalising features...")
    features_norm, feat_mean, feat_std = normalize_features(features, train_mask)

    # ── Pre-compute edge snapshots ─────────────────────────────────────────────
    print("\nPre-computing graph snapshots (monthly updates)...")
    edge_snapshots = precompute_edge_indices(
        returns        = returns,
        tickers        = valid_tickers,
        sectors        = sectors,
        valid_dates    = dates,
        corr_window    = config.CORR_WINDOW,
        corr_threshold = config.CORR_THRESHOLD,
        update_freq    = config.GRAPH_UPDATE_FREQ,
    )
    n_edges = edge_snapshots[-1][0].shape[1]
    print(f"  Graph has ~{n_edges} edges per snapshot")

    # ── Save everything ────────────────────────────────────────────────────────
    print("\nSaving processed data...")

    np.save(config.DATA_DIR + "processed/features.npy",      features_norm)
    np.save(config.DATA_DIR + "processed/targets.npy",       targets)
    np.save(config.DATA_DIR + "processed/fwd_returns.npy",   fwd_returns)
    np.save(config.DATA_DIR + "processed/feat_mean.npy",     feat_mean)
    np.save(config.DATA_DIR + "processed/feat_std.npy",      feat_std)
    np.save(config.DATA_DIR + "processed/train_indices.npy", train_indices)
    np.save(config.DATA_DIR + "processed/val_indices.npy",   val_indices)
    np.save(config.DATA_DIR + "processed/test_indices.npy",  test_indices)

    pd.Series(dates, name="date").to_frame().to_parquet(
        config.DATA_DIR + "processed/dates.parquet"
    )
    pd.Series(valid_tickers, name="ticker").to_frame().to_parquet(
        config.DATA_DIR + "processed/tickers.parquet"
    )

    with open(config.DATA_DIR + "processed/edge_snapshots.pkl", "wb") as f:
        pickle.dump(edge_snapshots, f)

    print("  Saved to data/processed/")
    print("Step 2 complete.\n")


if __name__ == "__main__":
    main()

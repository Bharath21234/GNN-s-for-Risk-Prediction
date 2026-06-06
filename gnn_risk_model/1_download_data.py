"""
Step 1 — Download and cache price data.

Fetches adjusted close prices for all 100 S&P 500 stocks plus SPY (market
proxy) and VIX from Yahoo Finance, computes log returns, and saves everything
to data/raw/.

Runtime: ~2–5 minutes depending on connection.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.data import download_prices, compute_log_returns

os.makedirs(config.DATA_DIR + "raw", exist_ok=True)


def main():
    # ── 1. Download stock prices ────────────────────────────────────────────────
    tickers = list(config.STOCKS.keys())
    prices  = download_prices(tickers, config.START_DATE, config.END_DATE)

    # Keep only tickers that downloaded successfully
    valid_tickers = list(prices.columns)
    if len(valid_tickers) < len(tickers):
        print(f"Proceeding with {len(valid_tickers)} tickers.")

    prices.to_parquet(config.DATA_DIR + "raw/prices.parquet")
    print(f"Saved prices: {prices.shape}  →  data/raw/prices.parquet")

    # ── 2. Compute log returns ──────────────────────────────────────────────────
    returns = compute_log_returns(prices)
    returns.to_parquet(config.DATA_DIR + "raw/returns.parquet")
    print(f"Saved returns: {returns.shape}  →  data/raw/returns.parquet")

    # ── 3. Download SPY (market benchmark) ─────────────────────────────────────
    spy = download_prices(["SPY"], config.START_DATE, config.END_DATE)
    spy_ret = compute_log_returns(spy)
    spy_ret.columns = ["SPY"]
    spy_ret.to_parquet(config.DATA_DIR + "raw/spy_returns.parquet")
    print(f"Saved SPY returns: {spy_ret.shape}")

    # ── 4. Download VIX ─────────────────────────────────────────────────────────
    vix_raw = yf.download("^VIX", start=config.START_DATE, end=config.END_DATE,
                           auto_adjust=True, progress=False)
    vix = vix_raw["Close"].squeeze().ffill().bfill()  # squeeze DataFrame→Series
    vix = vix.reindex(returns.index, method="ffill")
    vix.rename("VIX").to_frame().to_parquet(config.DATA_DIR + "raw/vix.parquet")
    print(f"Saved VIX: {vix.shape}")

    # ── 5. Summary stats ────────────────────────────────────────────────────────
    print("\n── Summary ────────────────────────────────────────────────────────────")
    print(f"Date range   : {returns.index[0].date()} → {returns.index[-1].date()}")
    print(f"Trading days : {len(returns)}")
    print(f"Stocks       : {len(returns.columns)}")
    print(f"Mean daily ret: {returns.mean().mean()*100:.3f}%")
    print(f"Mean daily vol: {returns.std().mean()*100:.3f}%")
    print("Step 1 complete.\n")


if __name__ == "__main__":
    main()

"""Data downloading, preprocessing, and feature engineering."""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    prices = prices.ffill().bfill()

    # Drop columns with more than 10% missing after fill
    threshold = 0.10 * len(prices)
    n_before = prices.shape[1]
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.90))
    n_after = prices.shape[1]
    if n_before != n_after:
        dropped = set(tickers) - set(prices.columns)
        print(f"  Dropped {n_before - n_after} tickers with too many NaNs: {dropped}")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns, forward-fill any residual NaNs."""
    rets = np.log(prices / prices.shift(1)).iloc[1:]
    return rets.ffill().bfill()


def compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    delta = returns.copy()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_node_features(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    vix: pd.Series,
    sectors: dict[str, str],
    all_sectors: list[str],
    t: int,
    lookback: int = 252,
) -> np.ndarray:
    """
    Build node feature matrix [N, F] for time step t (index into returns).

    Features per stock (27 total):
      0-3   : rolling vol (annualised) at 5, 10, 20, 60 days
      4-7   : rolling return at 1, 5, 20, 60 days
      8-9   : beta vs market at 20, 60 days
      10    : vol-of-vol (20-day std of 20-day rolling vol)
      11    : RSI-14
      12    : 20-day max drawdown
      13    : log(VIX) at time t
      14-24 : sector one-hot (11 sectors)
      25    : mean 60-day intra-sector correlation
      26    : 60-day correlation with market (SPY)
    """
    tickers = list(returns.columns)
    N = len(tickers)
    n_sectors = len(all_sectors)
    sector_to_idx = {s: i for i, s in enumerate(all_sectors)}

    start = max(0, t - lookback)
    window_ret = returns.iloc[start: t]    # shape (lookback, N)
    spy_win    = spy_returns.iloc[start: t]

    features = np.zeros((N, 14 + n_sectors + 2))  # 14 + 11 + 2 = 27

    # Precompute rolling stats for each stock
    WINDOWS = [5, 10, 20, 60]
    for i, ticker in enumerate(tickers):
        r = window_ret[ticker].values  # (lookback,)

        # rolling vol
        for j, w in enumerate(WINDOWS):
            if len(r) >= w:
                features[i, j] = float(np.std(r[-w:]) * np.sqrt(252))

        # rolling returns
        for j, w in enumerate([1, 5, 20, 60]):
            if len(r) >= w:
                features[i, 4 + j] = float(np.sum(r[-w:]))

        # beta vs market (OLS slope)
        spy_r = spy_win.values
        for j, w in enumerate([20, 60]):
            if len(r) >= w and len(spy_r) >= w:
                y, x = r[-w:], spy_r[-w:]
                cov = np.cov(x, y, ddof=1)
                features[i, 8 + j] = cov[0, 1] / (cov[0, 0] + 1e-12)

        # vol-of-vol: rolling 20-day vols over past 60 days
        if len(r) >= 60:
            vov_window = [np.std(r[k: k + 20]) for k in range(0, 40, 5) if k + 20 <= len(r)]
            if vov_window:
                features[i, 10] = float(np.std(vov_window) * np.sqrt(252))

        # RSI-14
        if len(r) >= 15:
            rsi_series = compute_rsi(pd.Series(r), window=14)
            features[i, 11] = float(rsi_series.iloc[-1])

        # 20-day max drawdown
        if len(r) >= 20:
            cum = np.exp(np.cumsum(r[-20:]))
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max) / (running_max + 1e-12)
            features[i, 12] = float(np.min(dd))

        # log(VIX)
        if t < len(vix):
            features[i, 13] = float(np.log(vix.iloc[t] + 1e-6))

        # sector one-hot
        sec = sectors.get(ticker, "")
        if sec in sector_to_idx:
            features[i, 14 + sector_to_idx[sec]] = 1.0

        # 60-day correlation with market
        if len(r) >= 60 and len(spy_r) >= 60:
            corr = np.corrcoef(r[-60:], spy_r[-60:])[0, 1]
            features[i, 14 + n_sectors + 1] = float(corr) if np.isfinite(corr) else 0.0

    # Mean intra-sector 60-day correlation (feature index 14+n_sectors)
    sector_groups: dict[str, list[int]] = {}
    for i, ticker in enumerate(tickers):
        sec = sectors.get(ticker, "other")
        sector_groups.setdefault(sec, []).append(i)

    if len(window_ret) >= 60:
        corr_matrix = window_ret.iloc[-60:].corr().values  # (N, N)
        for i, ticker in enumerate(tickers):
            sec = sectors.get(ticker, "")
            peers = [j for j in sector_groups.get(sec, []) if j != i]
            if peers:
                mean_corr = np.nanmean([corr_matrix[i, j] for j in peers])
                features[i, 14 + n_sectors] = float(mean_corr) if np.isfinite(mean_corr) else 0.0

    return features.astype(np.float32)


def compute_cvar_target(
    returns: pd.DataFrame,
    t: int,
    lookback: int = 252,
    alpha: float = 0.95,
    horizon: int = 5,
) -> np.ndarray:
    """
    Rolling historical CVaR at confidence level alpha, scaled to `horizon` days.

    At time t we use returns[t-lookback : t-1] (no lookahead).
    CVaR_1d = E[ -r | r <= q_{1-alpha} ]   (positive = expected loss)
    CVaR_5d = CVaR_1d * sqrt(horizon)        (square-root-of-time scaling)
    """
    start = max(0, t - lookback)
    window = returns.iloc[start: t].values  # (lookback, N)

    q = np.nanquantile(window, 1 - alpha, axis=0)  # (N,)
    cvar = np.zeros(window.shape[1])
    for i in range(window.shape[1]):
        tail = window[window[:, i] <= q[i], i]
        cvar[i] = -np.mean(tail) if len(tail) > 0 else 0.0

    return (cvar * np.sqrt(horizon)).astype(np.float32)


def compute_forward_returns(
    returns: pd.DataFrame,
    horizon: int = 5,
) -> np.ndarray:
    """
    Realised cumulative `horizon`-day forward log return for every (t, stock).

    fwd[t, i] = sum(returns[t : t+horizon, i])   (negative = a forward loss)

    The final `horizon-1` rows have no full forward window and are set to NaN.
    This is the *forward-looking* label used for FZ0/pinball training and
    evaluation (contrast with the backward-looking rolling CVaR target).
    """
    vals = returns.values                       # (T, N)
    T, N = vals.shape
    fwd = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        if t + horizon <= T:
            fwd[t] = vals[t: t + horizon].sum(axis=0)
    return fwd


def build_dataset(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    vix: pd.Series,
    sectors: dict[str, str],
    all_sectors: list[str],
    lookback: int = 252,
    alpha: float = 0.95,
    horizon: int = 5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build features, the backward-looking rolling-CVaR target (secondary
    reconstruction metric), and the forward-looking realised return label
    (primary training/evaluation signal), across all valid time steps.

    Returns
    -------
    features    : (T, N, F)
    targets     : (T, N)      backward-looking rolling CVaR (reconstruction only)
    fwd_returns : (T, N)      realised horizon-day forward return (NaN if run off)
    dates       : DatetimeIndex of length T
    """
    T = len(returns)
    N = len(returns.columns)
    F = 14 + len(all_sectors) + 2  # 27

    valid_start = lookback  # first t with full lookback
    T_valid = T - valid_start

    features = np.zeros((T_valid, N, F), dtype=np.float32)
    targets  = np.zeros((T_valid, N),    dtype=np.float32)

    it = range(valid_start, T)
    if verbose:
        it = tqdm(it, desc="Building features", total=T_valid)

    for idx, t in enumerate(it):
        features[idx] = compute_node_features(
            returns, spy_returns, vix, sectors, all_sectors, t, lookback
        )
        targets[idx] = compute_cvar_target(returns, t, lookback, alpha, horizon)

    fwd_all = compute_forward_returns(returns, horizon)   # (T, N)
    fwd_returns = fwd_all[valid_start:]                   # (T_valid, N)

    dates = returns.index[valid_start:]
    return features, targets, fwd_returns, dates


def normalize_features(
    features: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features using statistics from training time steps only.
    One-hot (binary) columns are left unchanged.

    Returns: (normalized_features, mean, std)
    """
    train_features = features[train_mask]                      # (T_train, N, F)
    flat = train_features.reshape(-1, train_features.shape[-1])  # (T_train*N, F)

    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8

    # Don't normalise binary/sector one-hot columns (they are already 0/1)
    n_sectors = len(config.SECTORS)
    binary_start = 14
    binary_end   = 14 + n_sectors
    mean[binary_start:binary_end] = 0.0
    std[binary_start:binary_end]  = 1.0

    norm_features = (features - mean) / std
    return norm_features.astype(np.float32), mean, std

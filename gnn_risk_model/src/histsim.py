"""
Historical-simulation (empirical) VaR / ES baseline.

The standard non-parametric tail-risk benchmark: at each date, form the
empirical distribution of *past* overlapping `horizon`-day returns over a
trailing window and read the VaR and ES straight off that distribution — no
distributional assumption, no fitting.  This is the natural non-parametric
counterpart to the parametric GARCH baselines and to the GNN.

Sign convention matches src.metrics: VaR/ES are returned as **positive loss
magnitudes** with ES >= VaR >= 0.

No look-ahead: for a forecast made at date d (predicting the return over the
next `horizon` days), only `horizon`-day returns that *end at or before d-1*
enter the empirical window — the same strictly-past information the GNN's
features use.
"""

import numpy as np
import pandas as pd


def _empirical_var_es(h_returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR/ES (positive losses) from a sample of h-day returns."""
    if len(h_returns) < 20:
        return np.nan, np.nan
    q = np.quantile(h_returns, alpha)          # alpha-quantile of returns (left tail)
    var_pos = max(-q, 1e-3)                    # matches the floor in src/metrics.py
    tail = h_returns[h_returns <= q]
    es_pos = -tail.mean() if len(tail) else var_pos
    es_pos = max(es_pos, var_pos)              # enforce ES >= VaR
    return float(var_pos), float(es_pos)


def run_histsim(
    returns: pd.DataFrame,
    pred_dates: pd.DatetimeIndex,
    tickers: list[str],
    alpha: float = 0.05,
    horizon: int = 5,
    window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling historical-simulation VaR/ES for every (date, ticker).

    Returns (var_df, es_df), both indexed by `pred_dates`, columns `tickers`,
    as positive loss magnitudes.
    """
    idx = returns.index
    var_df = pd.DataFrame(index=pred_dates, columns=tickers, dtype=float)
    es_df  = pd.DataFrame(index=pred_dates, columns=tickers, dtype=float)

    for tk in tickers:
        if tk not in returns.columns:
            continue
        r = returns[tk].values.astype(float)
        csum = np.concatenate([[0.0], np.cumsum(r)])   # csum[k] = sum r[:k]

        for d in pred_dates:
            pos = idx.searchsorted(d)
            # h-day returns ending at day s (0-based): csum[s+1]-csum[s+1-h].
            # Use only windows that end strictly before the forecast start (<= pos-1).
            s_hi = pos - 1
            s_lo = max(horizon - 1, pos - window)
            if s_hi < s_lo:
                continue
            s = np.arange(s_lo, s_hi + 1)
            hvals = csum[s + 1] - csum[s + 1 - horizon]
            v, e = _empirical_var_es(hvals, alpha)
            var_df.loc[d, tk] = v
            es_df.loc[d, tk]  = e

    return var_df.astype(float), es_df.astype(float)

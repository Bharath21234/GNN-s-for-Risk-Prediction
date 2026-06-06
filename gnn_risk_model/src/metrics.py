"""
Evaluation metrics for CVaR models.

Statistical tests implemented:
  - Kupiec (1995) Proportion of Failures (POF) test
  - Christoffersen (1998) Conditional Coverage (CC) test
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ── Point forecast accuracy ────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QLIKE loss (asymmetric, penalises under-prediction more).
    QLIKE = mean( y_true/y_pred - log(y_true/y_pred) - 1 )
    """
    ratio = y_true / (y_pred + 1e-8)
    return float(np.mean(ratio - np.log(ratio + 1e-8) - 1))


# ── Backtesting ────────────────────────────────────────────────────────────────

def _violation_sequence(returns_5d: np.ndarray, cvar_pred: np.ndarray) -> np.ndarray:
    """
    Return binary hit sequence: 1 if actual loss > predicted CVaR.
    `returns_5d` should be the actual 5-day return (negative = loss).
    """
    return (-returns_5d > cvar_pred).astype(int)


def kupiec_pof_test(
    violations: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures test.

    H0: E[violations] = alpha.

    Returns dict with LR statistic, p-value, and pass/fail at 5% significance.
    """
    T   = len(violations)
    N   = violations.sum()
    p   = N / T if T > 0 else 0.0

    if p == 0 or p == 1 or T == 0:
        return {"lr_stat": np.nan, "p_value": np.nan, "pass": False,
                "violation_rate": p, "n_violations": int(N), "n_obs": T}

    lr = -2 * (
        N * np.log(alpha / p) + (T - N) * np.log((1 - alpha) / (1 - p))
    )
    p_val = 1 - chi2.cdf(lr, df=1)

    return {
        "lr_stat":        float(lr),
        "p_value":        float(p_val),
        "pass":           bool(p_val > 0.05),  # fail to reject H0 = model correct
        "violation_rate": float(p),
        "n_violations":   int(N),
        "n_obs":          T,
    }


def christoffersen_cc_test(
    violations: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Christoffersen (1998) Conditional Coverage test.

    Tests both unconditional coverage (POF) and independence of violations.
    CC LR = LR_pof + LR_ind, both asymptotically χ²(1).
    """
    T = len(violations)
    if T < 2:
        return {"lr_stat": np.nan, "p_value": np.nan, "pass": False}

    v = violations
    n00 = ((v[:-1] == 0) & (v[1:] == 0)).sum()
    n01 = ((v[:-1] == 0) & (v[1:] == 1)).sum()
    n10 = ((v[:-1] == 1) & (v[1:] == 0)).sum()
    n11 = ((v[:-1] == 1) & (v[1:] == 1)).sum()

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p   = (n01 + n11) / (n00 + n01 + n10 + n11) if T > 0 else 0.0

    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    # LR independence
    lr_ind = -2 * (
        n00 * _safe_log(1 - p) + n01 * _safe_log(p)
        - n00 * _safe_log(1 - p01) - n01 * _safe_log(p01 + 1e-12)
        - n10 * _safe_log(1 - p11 + 1e-12) - n11 * _safe_log(p11 + 1e-12)
        + n10 * _safe_log(1 - p) + n11 * _safe_log(p + 1e-12)
    )

    # LR unconditional (POF part)
    N = v.sum()
    if N == 0 or N == T:
        lr_uc = np.nan
    else:
        lr_uc = -2 * (
            N * np.log(alpha / (N / T)) + (T - N) * np.log((1 - alpha) / (1 - N / T))
        )

    lr_cc = (lr_ind if np.isfinite(lr_ind) else 0) + (lr_uc if np.isfinite(lr_uc) else 0)
    p_val = 1 - chi2.cdf(lr_cc, df=2) if np.isfinite(lr_cc) else np.nan

    return {
        "lr_stat":  float(lr_cc),
        "p_value":  float(p_val) if np.isfinite(p_val) else np.nan,
        "pass":     bool(p_val > 0.05) if np.isfinite(p_val) else False,
        "pi01":     float(p01),
        "pi11":     float(p11),
    }


def backtest_model(
    cvar_pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Run Kupiec and Christoffersen tests for each stock and aggregate.

    `cvar_pred_df` : (T, N) predicted 5-day CVaR (positive = risk level)
    `returns_df`   : (T_full, N) daily log returns (aligned dates)
    """
    dates   = cvar_pred_df.index
    tickers = cvar_pred_df.columns

    # Compute realised 5-day forward returns for each date in cvar_pred_df
    returns_aligned = returns_df.reindex(returns_df.index.union(dates))

    kupiec_results = {}
    cc_results     = {}

    for ticker in tickers:
        preds = cvar_pred_df[ticker].values          # (T,)
        viols = []

        for i, d in enumerate(dates):
            pos = returns_df.index.searchsorted(d)
            if pos + horizon > len(returns_df):
                viols.append(np.nan)
                continue
            fwd_return = returns_df[ticker].iloc[pos: pos + horizon].sum()
            viols.append(1 if (-fwd_return > preds[i]) else 0)

        viols_arr = np.array(viols)
        mask      = np.isfinite(viols_arr)
        viols_arr = viols_arr[mask].astype(int)
        preds_cut = preds[mask]

        kupiec_results[ticker] = kupiec_pof_test(viols_arr, alpha=alpha)
        cc_results[ticker]     = christoffersen_cc_test(viols_arr, alpha=alpha)

    # Aggregate across stocks
    viol_rates = [v["violation_rate"] for v in kupiec_results.values()]
    pof_passes = [v["pass"] for v in kupiec_results.values()]
    cc_passes  = [v["pass"] for v in cc_results.values()]

    return {
        "kupiec":  kupiec_results,
        "cc":      cc_results,
        "summary": {
            "mean_violation_rate": float(np.mean(viol_rates)),
            "kupiec_pass_pct":     float(np.mean(pof_passes) * 100),
            "cc_pass_pct":         float(np.mean(cc_passes) * 100),
        },
    }


def compute_accuracy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    """Compute all point-forecast accuracy metrics."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "model":  model_name,
        "rmse":   rmse(yt, yp),
        "mae":    mae(yt, yp),
        "mape":   mape(yt, yp),
        "qlike":  qlike(yt, yp),
        "n_obs":  int(mask.sum()),
    }

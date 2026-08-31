"""
Evaluation metrics for VaR / ES (CVaR) models.

Sign / storage convention used throughout this module
-----------------------------------------------------
* Returns are (log) returns; a loss is a *negative* return.
* `var_pos` and `es_pos` are stored as **positive loss magnitudes**
  (the amount you expect to lose), with `es_pos >= var_pos >= 0`.
  In return-quantile space they correspond to v = -var_pos and e = -es_pos,
  both <= 0.
* A "violation" (breach) at the 95% level is `-realised_return > var_pos`,
  i.e. the realised loss exceeds the predicted VaR.

Backtests / scores implemented
------------------------------
Forward-looking accuracy (scored vs *realised* forward returns):
  - pinball_loss           : quantile (VaR) loss at level alpha
  - fz0_loss               : Fissler-Ziegel (0-homogeneous) joint (VaR, ES) loss
                             — strictly consistent for (VaR, ES) [PZC 2019]
VaR calibration (now correctly specified — expected breach rate = alpha):
  - kupiec_pof_test        : Kupiec (1995) Proportion-of-Failures
  - christoffersen_cc_test : Christoffersen (1998) Conditional Coverage
ES calibration:
  - acerbi_szekely_z2      : Acerbi-Szekely (2014) Test 2 statistic
  - es_residual_test       : t-test that mean exceedance residual (-r/ES) = 1
Significance of forecast-loss differences:
  - diebold_mariano        : DM test with Newey-West HAC variance and the
                             Harvey-Leybourne-Newbold small-sample correction
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t, norm


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


# ════════════════════════════════════════════════════════════════════════════
# Forward-looking, strictly-consistent scoring for (VaR, ES)
# ════════════════════════════════════════════════════════════════════════════

def realized_forward_returns(
    returns_df: pd.DataFrame,
    pred_dates: pd.DatetimeIndex,
    tickers: list[str],
    horizon: int = 5,
) -> np.ndarray:
    """
    Cumulative realised `horizon`-day forward log return for each (date, ticker)
    in a prediction frame.  Returns array shape (len(pred_dates), len(tickers));
    entries where the full horizon runs past the end of `returns_df` are NaN.
    """
    idx = returns_df.index
    out = np.full((len(pred_dates), len(tickers)), np.nan, dtype=float)
    cols = [returns_df.columns.get_loc(tk) for tk in tickers]
    vals = returns_df.values
    for i, d in enumerate(pred_dates):
        pos = idx.searchsorted(d)
        if pos + horizon > len(idx):
            continue
        out[i] = vals[pos: pos + horizon, cols].sum(axis=0)
    return out


def pinball_loss(
    realised: np.ndarray,
    var_pos: np.ndarray,
    alpha: float = 0.05,
    reduce: bool = True,
):
    """
    Quantile (pinball) loss for the VaR forecast at tail level `alpha`.

    We predict the alpha-quantile of returns, q = -var_pos (a loss level).
    L = (r - q)(alpha - 1{r < q}); lower is better.  `reduce=False` returns the
    per-observation series (needed for Diebold-Mariano).
    """
    r  = np.asarray(realised, dtype=float)
    q  = -np.asarray(var_pos, dtype=float)
    ind = (r < q).astype(float)
    loss = (r - q) * (alpha - ind)
    return float(np.nanmean(loss)) if reduce else loss


def fz0_loss(
    realised: np.ndarray,
    var_pos: np.ndarray,
    es_pos: np.ndarray,
    alpha: float = 0.05,
    reduce: bool = True,
):
    """
    Fissler-Ziegel FZ0 loss — the 0-homogeneous, strictly consistent joint
    scoring function for (VaR, ES) of Patton, Ziegel & Chen (2019).

    In return-quantile space (v = -var_pos, e = -es_pos, both < 0):

        L = -1/(alpha*e) * 1{r <= v} * (v - r) + v/e + log(-e) - 1

    Rewritten with positive loss magnitudes (es_pos >= var_pos > 0):

        L = 1{-r >= var_pos} * (-var_pos - r)/(alpha*es_pos)
            + var_pos/es_pos + log(es_pos) - 1

    Lower is better.  Requires es_pos > 0.
    """
    r  = np.asarray(realised, dtype=float)
    vp = np.asarray(var_pos,  dtype=float)
    # Floor well below the model's own 5e-3 ES floor (see gnn_model.py) so it
    # never binds there; guards other baselines against a near-zero ES
    # blowing up the 1/es_pos terms below (1e-8 was too permissive).
    ep = np.clip(np.asarray(es_pos, dtype=float), 1e-3, None)  # ES must be > 0
    breach = (-r >= vp).astype(float)
    loss = breach * (-vp - r) / (alpha * ep) + vp / ep + np.log(ep) - 1.0
    return float(np.nanmean(loss)) if reduce else loss


# ════════════════════════════════════════════════════════════════════════════
# ES-specific calibration tests
# ════════════════════════════════════════════════════════════════════════════

def acerbi_szekely_z2(
    realised: np.ndarray,
    var_pos: np.ndarray,
    es_pos: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """
    Acerbi-Szekely (2014) Test 2 for expected shortfall.

        Z2 = 1 + (1 / (T*alpha)) * sum_t [ r_t * 1{-r_t > var_pos} / es_pos_t ]

    H0 (ES correctly specified) => E[Z2] = 0.  Z2 < 0 means realised tail losses
    exceed the predicted ES (risk under-estimated); Z2 > 0 means over-estimated.
    A one-sided bootstrap p-value (resampling the per-obs contributions) is
    reported for H0: Z2 >= 0 vs risk under-estimation.
    """
    r  = np.asarray(realised, dtype=float)
    vp = np.asarray(var_pos,  dtype=float)
    # Floor well below the model's own 5e-3 ES floor (see gnn_model.py) so it
    # never binds there; guards other baselines against a near-zero ES
    # blowing up the 1/es_pos terms below (1e-8 was too permissive).
    ep = np.clip(np.asarray(es_pos, dtype=float), 1e-3, None)
    mask = np.isfinite(r) & np.isfinite(vp) & np.isfinite(ep)
    r, vp, ep = r[mask], vp[mask], ep[mask]
    T = len(r)
    if T == 0:
        return {"z2": np.nan, "p_value": np.nan, "n_breaches": 0, "n_obs": 0}

    breach = (-r > vp).astype(float)
    contrib = r * breach / ep                      # per-obs contribution
    z2 = 1.0 + contrib.sum() / (T * alpha)

    # Bootstrap the sampling distribution of Z2 under resampling of contributions
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        samp = contrib[rng.integers(0, T, T)]
        boot[b] = 1.0 + samp.sum() / (T * alpha)
    # one-sided: probability of a Z2 as low (risk under-estimated) as observed
    p_val = float(np.mean(boot <= 0.0)) if z2 < 0 else float(np.mean(boot >= 0.0))

    return {
        "z2":         float(z2),
        "p_value":    p_val,
        "n_breaches": int(breach.sum()),
        "n_obs":      int(T),
    }


def es_residual_test(
    realised: np.ndarray,
    var_pos: np.ndarray,
    es_pos: np.ndarray,
) -> dict:
    """
    Conditional ES-calibration check on the exceedance residuals.

    On breaches (-r > var_pos), the standardised residual rho = (-r)/es_pos
    should have mean 1 if the ES level is correct.  Two-sided t-test of
    H0: mean(rho) = 1.  mean(rho) > 1 => ES too small (under-estimates tail).
    """
    r  = np.asarray(realised, dtype=float)
    vp = np.asarray(var_pos,  dtype=float)
    # Floor well below the model's own 5e-3 ES floor (see gnn_model.py) so it
    # never binds there; guards other baselines against a near-zero ES
    # blowing up the 1/es_pos terms below (1e-8 was too permissive).
    ep = np.clip(np.asarray(es_pos, dtype=float), 1e-3, None)
    mask = np.isfinite(r) & np.isfinite(vp) & np.isfinite(ep) & (-r > vp)
    rho = (-r[mask]) / ep[mask]
    n = len(rho)
    if n < 2:
        return {"mean_ratio": float(np.mean(rho)) if n else np.nan,
                "t_stat": np.nan, "p_value": np.nan, "n_breaches": n}
    se = rho.std(ddof=1) / np.sqrt(n)
    tstat = (rho.mean() - 1.0) / se if se > 0 else np.nan
    p_val = float(2 * (1 - student_t.cdf(abs(tstat), df=n - 1))) if np.isfinite(tstat) else np.nan
    return {
        "mean_ratio": float(rho.mean()),
        "t_stat":     float(tstat) if np.isfinite(tstat) else np.nan,
        "p_value":    p_val,
        "n_breaches": int(n),
    }


# ════════════════════════════════════════════════════════════════════════════
# Diebold-Mariano test for equal forecast accuracy
# ════════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 5,
) -> dict:
    """
    Diebold-Mariano (1995) test of equal predictive accuracy on two per-period
    loss series (same scoring function).  d_t = loss_a - loss_b.

    Uses a Newey-West (HAC) long-run variance with lag = horizon - 1 to handle
    the overlap induced by the `horizon`-day forecast, and the
    Harvey-Leybourne-Newbold (1997) small-sample correction; p-value from a
    Student-t with T-1 df.

    Negative DM statistic => model A has lower loss (A is better).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 3:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan, "n_obs": T}

    d_bar = d.mean()
    dc = d - d_bar
    L = max(0, horizon - 1)
    gamma0 = np.mean(dc * dc)
    lrv = gamma0
    for lag in range(1, L + 1):
        if lag >= T:
            break
        w = 1.0 - lag / (L + 1)                    # Bartlett kernel
        gamma = np.mean(dc[lag:] * dc[:-lag])
        lrv += 2.0 * w * gamma

    if lrv <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": float(d_bar), "n_obs": T}

    dm = d_bar / np.sqrt(lrv / T)
    # Harvey-Leybourne-Newbold small-sample correction
    h = horizon
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * corr
    p_val = float(2 * (1 - student_t.cdf(abs(dm_star), df=T - 1)))
    return {
        "dm_stat":   float(dm_star),
        "p_value":   p_val,
        "mean_diff": float(d_bar),
        "n_obs":     int(T),
    }


def moving_block_bootstrap_ci(
    loss_series: np.ndarray,
    block: int = 5,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Moving-block-bootstrap confidence interval for the mean of a (serially
    dependent) per-period loss series.  The block length preserves the
    autocorrelation induced by the overlapping `horizon`-day forecast.

    Returns the point mean and the (lo, hi) percentile CI.
    """
    x = np.asarray(loss_series, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < block + 1:
        m = float(np.mean(x)) if T else np.nan
        return {"mean": m, "lo": np.nan, "hi": np.nan, "n_obs": T}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts_max = T - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T]
        means[b] = x[idx].mean()
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n_obs": T}

"""
GARCH baselines for 5-day CVaR prediction.

Baseline 1 (ours):  GARCH(1,1)
    Bollerslev, T. (1986). Generalised autoregressive conditional
    heteroscedasticity. Journal of Econometrics, 31(3), 307-327.

Baseline 2 (paper): GJR-GARCH(1,1)
    Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). On the relation
    between the expected value and the volatility of the nominal excess return
    on stocks. Journal of Finance, 48(5), 1779-1801.

    GJR-GARCH adds an asymmetric "leverage" term γ so that negative shocks
    (σ²_{t} = ω + (α + γ·I_{t-1})·r²_{t-1} + β·σ²_{t-1}) increase
    conditional variance more than positive shocks of the same magnitude.
    This captures the well-documented leverage effect in equity returns.
"""

import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from tqdm import tqdm


# ── VaR / ES formulae for a normal conditional distribution ────────────────────
_Z005       = norm.ppf(0.05)    # ≈ -1.6449
_PHI_Z005   = norm.pdf(_Z005)   # ≈  0.1031
_VAR_SCALE  = -_Z005            # ≈  1.6449  (= 95% VaR of a standard normal)
_CVAR_SCALE = _PHI_Z005 / 0.05  # ≈  2.0627  (= E[-Z | Z < z_{0.05}])


def _parametric_var(sigma_1d: float, horizon: int = 5) -> float:
    """95% VaR for a normal(0, sigma) daily return, √-time scaled. Positive=loss."""
    return float(sigma_1d * np.sqrt(horizon) * _VAR_SCALE)


def _parametric_cvar(sigma_1d: float, horizon: int = 5) -> float:
    """95% CVaR/ES for a normal(0, sigma) daily return, √-time scaled. Positive=loss."""
    return float(sigma_1d * np.sqrt(horizon) * _CVAR_SCALE)


def _fit_garch(returns: np.ndarray, model_type: str = "GARCH") -> object:
    """Fit GARCH(1,1) or GJR-GARCH(1,1) using arch library, suppress warnings."""
    # Scale returns to percentage (arch library convention)
    r = returns * 100
    if model_type == "GJR":
        am = arch_model(r, vol="GARCH", p=1, o=1, q=1, dist="normal")
    else:
        am = arch_model(r, vol="GARCH", p=1, o=0, q=1, dist="normal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    return res


def _forecast_sigma(res, horizon: int = 1) -> float:
    """1-step-ahead conditional std from fitted arch model (in original scale)."""
    fc = res.forecast(horizon=horizon, reindex=False)
    variance_pct_sq = fc.variance.iloc[-1, 0]  # variance in (pct)^2
    # Convert back: sigma_pct -> sigma_return
    return float(np.sqrt(variance_pct_sq) / 100.0)


def run_garch_rolling(
    returns: pd.DataFrame,
    test_start_date: str,
    refit_freq: int = 21,
    horizon: int = 5,
    model_type: str = "GARCH",
) -> pd.DataFrame:
    """
    Walk-forward CVaR predictions for all stocks using GARCH(1,1) or GJR-GARCH.

    Expanding window: re-fits every `refit_freq` trading days using all data up
    to the refit point, then uses the fitted model to forecast σ for the
    subsequent `refit_freq` days.

    Parameters
    ----------
    returns      : DataFrame of log returns (all dates, all stocks)
    test_start_date : first date of the test/evaluation window
    refit_freq   : trading days between model refits
    horizon      : CVaR horizon in days
    model_type   : "GARCH" or "GJR"

    Returns
    -------
    (var_df, es_df) : two DataFrames of predicted 95% VaR and ES (CVaR), both
    positive loss magnitudes; index=dates (test window), columns=tickers.
    """
    test_start_idx = returns.index.searchsorted(pd.Timestamp(test_start_date))
    test_dates     = returns.index[test_start_idx:]
    tickers        = list(returns.columns)

    var_pred = pd.DataFrame(index=test_dates, columns=tickers, dtype=float)
    es_pred  = pd.DataFrame(index=test_dates, columns=tickers, dtype=float)

    for ticker in tqdm(tickers, desc=f"  {model_type} baseline"):
        r     = returns[ticker].values
        dates = returns.index

        last_res   = None
        last_refit = -refit_freq  # force initial fit

        for t_abs in range(test_start_idx, len(returns)):
            # Re-fit on an expanding window every refit_freq days
            if t_abs - last_refit >= refit_freq:
                try:
                    last_res   = _fit_garch(r[:t_abs], model_type)
                    last_refit = t_abs
                except Exception:
                    last_res = None

            if last_res is None:
                # Fallback: historical volatility
                window = r[max(0, t_abs - 252): t_abs]
                sigma  = float(np.std(window)) if len(window) > 1 else 0.01
            else:
                try:
                    sigma = _forecast_sigma(last_res, horizon=1)
                except Exception:
                    window = r[max(0, t_abs - 252): t_abs]
                    sigma  = float(np.std(window)) if len(window) > 1 else 0.01

            var_pred.loc[dates[t_abs], ticker] = _parametric_var(sigma, horizon)
            es_pred.loc[dates[t_abs],  ticker] = _parametric_cvar(sigma, horizon)

    return var_pred.astype(float), es_pred.astype(float)


def run_garch(returns: pd.DataFrame, test_start_date: str, horizon: int = 5):
    """Convenience wrapper for GARCH(1,1). Returns (var_df, es_df)."""
    return run_garch_rolling(returns, test_start_date, model_type="GARCH", horizon=horizon)


def run_gjr_garch(returns: pd.DataFrame, test_start_date: str, horizon: int = 5):
    """Convenience wrapper for GJR-GARCH(1,1). Returns (var_df, es_df)."""
    return run_garch_rolling(returns, test_start_date, model_type="GJR", horizon=horizon)

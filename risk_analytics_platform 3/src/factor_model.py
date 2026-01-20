"""Simple two-factor model for stress testing.

Implements a lightweight, interview-friendly factor model:

  - Equity factor: SPY daily returns (market proxy)
  - Rate factor: daily change in 10Y yield (FRED DGS10)

For each asset, we estimate betas via OLS:

    r_i(t) = a_i + b_eq,i * f_eq(t) + b_rate,i * f_rate(t) + eps

Then we can run "factor shocks" (e.g., equity crash + rate hike) to produce
predicted asset returns and portfolio P&L.

No heavy dependencies (statsmodels) are used – just numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_factors(
    equity_factor_returns: pd.Series,
    rate_series: pd.Series,
) -> pd.DataFrame:
    """Build a factor dataframe aligned on date.

    Parameters
    ----------
    equity_factor_returns
        Daily returns for the equity market proxy (e.g. SPY).
    rate_series
        10Y yield level series (percentage, e.g. 4.2). We'll convert to a
        daily *yield change* in decimal terms.
    """
    if equity_factor_returns.empty or rate_series.empty:
        return pd.DataFrame()

    eq = equity_factor_returns.copy().astype(float)
    eq.name = "f_equity"

    y = rate_series.copy().astype(float) / 100.0  # percent -> decimal
    dy = y.diff()
    dy.name = "f_rate"

    fac = pd.concat([eq, dy], axis=1).dropna(how="any")
    return fac


def estimate_betas_ols(
    returns_wide: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate per-asset factor betas using OLS.

    Returns a dataframe with columns: alpha, beta_equity, beta_rate, r2.
    """
    if returns_wide.empty or factors.empty:
        return pd.DataFrame()

    # Align on common dates
    common_idx = returns_wide.index.intersection(factors.index)
    if len(common_idx) < 60:
        # not enough data for a stable regression
        return pd.DataFrame()

    Y = returns_wide.loc[common_idx].astype(float).values  # T x N
    X = factors.loc[common_idx][["f_equity", "f_rate"]].astype(float).values  # T x 2

    # add intercept
    ones = np.ones((X.shape[0], 1))
    X_ = np.hstack([ones, X])  # T x 3

    # OLS: B = (X'X)^-1 X'Y
    XtX = X_.T @ X_
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(XtX)

    B = inv @ X_.T @ Y  # 3 x N

    # R^2 per asset
    Y_hat = X_ @ B
    ss_res = ((Y - Y_hat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (ss_res / np.where(ss_tot == 0, np.nan, ss_tot))

    out = pd.DataFrame(
        {
            "ticker": returns_wide.columns,
            "alpha": B[0, :],
            "beta_equity": B[1, :],
            "beta_rate": B[2, :],
            "r2": r2,
        }
    ).set_index("ticker")

    return out


def factor_shock_pnl(
    weights: pd.Series,
    betas: pd.DataFrame,
    equity_shock: float,
    rate_shock_bps: float,
    include_alpha: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Compute portfolio P&L under a factor shock.

    equity_shock
        Shock to equity factor in return terms (e.g., -0.30 for -30%).
    rate_shock_bps
        Shock to 10Y yield in basis points (e.g., +200).

    Returns
    -------
    portfolio_pnl_pct, detail_df
    """
    if betas.empty:
        return float("nan"), pd.DataFrame()

    w = weights.reindex(betas.index).astype(float)
    w = w / w.sum()

    d_rate = float(rate_shock_bps) / 10000.0

    pred = (
        betas["beta_equity"] * float(equity_shock)
        + betas["beta_rate"] * float(d_rate)
    )
    if include_alpha:
        pred = pred + betas["alpha"]

    contrib = w * pred
    port = float(contrib.sum())

    detail = pd.DataFrame(
        {
            "weight": w,
            "predicted_return": pred,
            "pnl_contribution": contrib,
            "beta_equity": betas["beta_equity"],
            "beta_rate": betas["beta_rate"],
            "r2": betas["r2"],
        }
    ).reset_index().rename(columns={"index": "ticker"})
    detail = detail.sort_values("pnl_contribution")
    return port, detail

"""Risk analytics engine.

Implements:
- Portfolio returns
- Volatility (annualised)
- VaR (Historical, Parametric/Normal, Monte Carlo)
- Expected Shortfall
- Rolling volatility (for volatility clustering)

All calculations are performed in pandas/numpy for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS = 252


@dataclass
class RiskMetrics:
    vol_annualised: float
    var_hist: float
    es_hist: float
    var_parametric: float
    es_parametric: float
    var_mc: float
    es_mc: float


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from adjusted close.

    price_df columns: date, ticker, adj_close
    """
    df = price_df.copy()
    df = df.sort_values(["ticker", "date"])
    df["adj_close"] = df["adj_close"].astype(float)
    df["return"] = df.groupby("ticker")["adj_close"].pct_change()
    return df.dropna(subset=["return"])


def pivot_returns(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot returns into wide matrix: index=date, columns=ticker."""
    wide = ret_df.pivot(index="date", columns="ticker", values="return")
    # drop days with missing values for any asset for simplicity
    wide = wide.dropna(how="any")
    return wide


def portfolio_returns(returns_wide: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(returns_wide.columns).astype(float)
    weights = weights / weights.sum()
    port = returns_wide.mul(weights, axis=1).sum(axis=1)
    port.name = "portfolio_return"
    return port


def annualised_volatility(port_rets: pd.Series) -> float:
    return float(port_rets.std(ddof=1) * np.sqrt(TRADING_DAYS))


def var_historical(port_rets: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR as positive loss number."""
    q = np.quantile(port_rets, alpha)
    return float(-q)


def es_historical(port_rets: pd.Series, alpha: float = 0.05) -> float:
    cutoff = np.quantile(port_rets, alpha)
    tail = port_rets[port_rets <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def var_parametric_normal(port_rets: pd.Series, alpha: float = 0.05) -> float:
    mu = float(port_rets.mean())
    sigma = float(port_rets.std(ddof=1))
    z = norm.ppf(alpha)
    return float(-(mu + z * sigma))


def es_parametric_normal(port_rets: pd.Series, alpha: float = 0.05) -> float:
    """ES under normal distribution (closed form)."""
    mu = float(port_rets.mean())
    sigma = float(port_rets.std(ddof=1))
    z = norm.ppf(alpha)
    # ES formula for normal distribution
    es = mu - sigma * (norm.pdf(z) / alpha)
    return float(-es)


def var_monte_carlo(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo VaR/ES using multivariate normal approximation.

    returns_wide: daily returns matrix.
    weights: portfolio weights.
    """
    rng = np.random.default_rng(random_seed)

    mu_vec = returns_wide.mean().values
    cov = returns_wide.cov().values

    sims = rng.multivariate_normal(mean=mu_vec, cov=cov, size=n_sims)
    w = weights.reindex(returns_wide.columns).astype(float).values
    w = w / w.sum()

    port_sims = sims @ w
    var = float(-np.quantile(port_sims, alpha))
    tail = port_sims[port_sims <= np.quantile(port_sims, alpha)]
    es = float(-tail.mean()) if len(tail) else float("nan")

    return var, es


def rolling_volatility(port_rets: pd.Series, window: int = 30) -> pd.Series:
    rv = port_rets.rolling(window=window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    rv.name = f"rolling_vol_{window}d"
    return rv


def compute_all_metrics(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10000,
) -> RiskMetrics:
    port = portfolio_returns(returns_wide, weights)

    vol = annualised_volatility(port)
    v_hist = var_historical(port, alpha=alpha)
    e_hist = es_historical(port, alpha=alpha)

    v_para = var_parametric_normal(port, alpha=alpha)
    e_para = es_parametric_normal(port, alpha=alpha)

    v_mc, e_mc = var_monte_carlo(returns_wide, weights, alpha=alpha, n_sims=n_sims)

    return RiskMetrics(
        vol_annualised=vol,
        var_hist=v_hist,
        es_hist=e_hist,
        var_parametric=v_para,
        es_parametric=e_para,
        var_mc=v_mc,
        es_mc=e_mc,
    )

"""Risk attribution helpers.

These functions make the project feel closer to a "real" risk desk:

- Component VaR (Parametric / variance-covariance)
- ES contributions (Historical tail decomposition)
- Volatility contributions (Euler / covariance-based)

All outputs are in *return space* (daily), expressed as positive loss
numbers where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _clean_weights(returns_wide: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    w = weights.reindex(returns_wide.columns).astype(float).values
    w = w / np.sum(w)
    return w


def component_var_parametric(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Component VaR allocation for a normal / variance-covariance VaR.

    VaR ≈ -z * sigma_p, where z = norm.ppf(alpha) (negative), sigma_p is the
    portfolio standard deviation.

    Component VaR_i = w_i * ( (Sigma w)_i / sigma_p ) * (-z)

    Returns a dataframe that sums to total VaR.
    """
    if returns_wide.empty:
        return pd.DataFrame()

    cov = returns_wide.cov().values
    w = _clean_weights(returns_wide, weights)

    sigma_p = float(np.sqrt(w.T @ cov @ w))
    if sigma_p == 0:
        return pd.DataFrame()

    mrc = (cov @ w) / sigma_p  # marginal risk contribution to sigma
    z = float(norm.ppf(alpha))
    total_var = float((-z) * sigma_p)
    comp = w * mrc * (-z)

    out = pd.DataFrame(
        {
            "ticker": returns_wide.columns,
            "weight": w,
            "component_var": comp,
        }
    )
    out["pct_of_total"] = out["component_var"] / total_var
    out = out.sort_values("component_var", ascending=False)
    return out


def volatility_contribution(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """Volatility contribution by asset.

    Uses Euler decomposition: RC_i = w_i * (Sigma w)_i / sigma_p
    Summation equals portfolio volatility (daily).
    """
    if returns_wide.empty:
        return pd.DataFrame()

    cov = returns_wide.cov().values
    w = _clean_weights(returns_wide, weights)

    sigma_p = float(np.sqrt(w.T @ cov @ w))
    if sigma_p == 0:
        return pd.DataFrame()

    mrc = (cov @ w) / sigma_p
    rc = w * mrc

    out = pd.DataFrame(
        {"ticker": returns_wide.columns, "weight": w, "vol_contribution": rc}
    )
    out["pct_of_total"] = out["vol_contribution"] / sigma_p
    out = out.sort_values("vol_contribution", ascending=False)
    return out


def es_contribution_historical(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Historical ES contributions (tail average decomposition).

    Since portfolio return is the weighted sum of asset returns, ES can be
    decomposed additively over tail observations:

        ES = -mean( sum_i w_i r_{i,t} | portfolio in alpha-tail )
           = sum_i -mean( w_i r_{i,t} | tail )

    Returns contributions that sum (approximately) to ES.
    """
    if returns_wide.empty:
        return pd.DataFrame()

    w = _clean_weights(returns_wide, weights)
    port = returns_wide.values @ w
    cutoff = np.quantile(port, alpha)
    tail_mask = port <= cutoff

    if tail_mask.sum() == 0:
        return pd.DataFrame()

    tail_rets = returns_wide.loc[tail_mask]
    contrib = -tail_rets.mul(w, axis=1).mean(axis=0)
    total_es = float(contrib.sum())

    out = pd.DataFrame(
        {
            "ticker": contrib.index,
            "weight": w,
            "es_contribution": contrib.values,
        }
    )
    out["pct_of_total"] = out["es_contribution"] / total_es
    out = out.sort_values("es_contribution", ascending=False)
    return out


def component_var_historical(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Component VaR allocation for Historical VaR.

    Historical VaR is driven by an observed return scenario. A practical,
    interview-friendly decomposition is:

      1) Compute portfolio returns r_p(t) = sum_i w_i r_i(t)
      2) Find the VaR threshold q = quantile(r_p, alpha)
      3) Identify the scenario day closest to q
      4) Allocate VaR by each asset's loss on that scenario:

          VaR_i = -w_i * r_i(t*)

    This ensures the contributions sum to the portfolio loss on the VaR day.
    """
    if returns_wide.empty:
        return pd.DataFrame()

    w = _clean_weights(returns_wide, weights)
    port = returns_wide.values @ w

    q = float(np.quantile(port, alpha))
    idx = int(np.argmin(np.abs(port - q)))
    scenario_rets = returns_wide.iloc[idx]

    contrib = -(scenario_rets.values * w)
    total_var = float(-port[idx])
    if total_var == 0:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "ticker": returns_wide.columns,
            "weight": w,
            "component_var": contrib,
            "scenario_date": returns_wide.index[idx],
        }
    )
    out["pct_of_total"] = out["component_var"] / total_var
    out = out.sort_values("component_var", ascending=False)
    return out


def component_var_monte_carlo(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Component VaR allocation for Monte Carlo VaR.

    We simulate multivariate-normal scenarios consistent with the sample mean
    and covariance of asset returns (same assumption as the MC VaR itself).

    Allocation approach:
      - Find the simulated scenario closest to the VaR quantile
      - Allocate VaR by each asset's weighted loss in that scenario

    This mirrors the historical VaR decomposition (scenario-based), but in the
    Monte Carlo scenario space.
    """
    if returns_wide.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    mu_vec = returns_wide.mean().values
    cov = returns_wide.cov().values
    sims = rng.multivariate_normal(mean=mu_vec, cov=cov, size=int(n_sims))

    w = _clean_weights(returns_wide, weights)
    port_sims = sims @ w
    q = float(np.quantile(port_sims, alpha))
    idx = int(np.argmin(np.abs(port_sims - q)))

    contrib = -(sims[idx, :] * w)
    total_var = float(-port_sims[idx])
    if total_var == 0:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "ticker": returns_wide.columns,
            "weight": w,
            "component_var": contrib,
        }
    )
    out["pct_of_total"] = out["component_var"] / total_var
    out = out.sort_values("component_var", ascending=False)
    return out

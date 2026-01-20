"""Stress testing scenarios.

We implement two interview-friendly scenarios:
1) 2008-style equity crash
2) Interest rate hike (duration-based approximation)

All outputs are expressed as portfolio % P&L (negative is loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factor_model import estimate_betas_ols, factor_shock_pnl


@dataclass
class ScenarioResult:
    name: str
    shock_description: str
    portfolio_pnl_pct: float


@dataclass
class ScenarioDetail:
    """Per-position detail output for stress tests."""

    ticker: str
    asset_class: str
    weight: float
    dv01: float
    pnl_value: float
    pnl_pct: float


DEFAULT_DURATIONS = {
    "TLT": 17.0,  # long-duration US Treasuries ETF (approx)
    "IEF": 7.0,
    "LQD": 8.5,
}


def scenario_equity_crash(
    positions: pd.DataFrame,
    equity_shock: float = -0.30,
    fi_shock: float = -0.10,
) -> ScenarioResult:
    """Apply a simple one-step crash shock by asset class."""
    df = positions.copy()
    df["asset_class"] = df["asset_class"].fillna("equity")

    shocks = np.where(df["asset_class"].str.lower().str.contains("fixed|bond|fi"), fi_shock, equity_shock)
    pnl = float((df["weight"].astype(float) * shocks).sum())

    return ScenarioResult(
        name="2008-style Crash",
        shock_description=f"Equities {equity_shock:.0%}, Fixed Income {fi_shock:.0%}",
        portfolio_pnl_pct=pnl,
    )


def scenario_rate_hike(
    positions: pd.DataFrame,
    rate_hike_bps: float = 100,
    equity_shock: float = -0.05,
    durations: dict | None = None,
) -> ScenarioResult:
    """Approximate rate-hike impact using duration: dP/P ~= -Duration * dY.

    rate_hike_bps: size of shock in basis points (e.g., 100 = +1.00%)
    equity_shock: optional equity impact during rapid hikes
    durations: mapping ticker->duration (years)

    If an asset is classed as fixed income, we use duration mapping if available.
    Otherwise we use a conservative fallback.
    """
    df = positions.copy()
    df["asset_class"] = df["asset_class"].fillna("equity")

    dy = float(rate_hike_bps) / 10000.0
    dur_map = durations or DEFAULT_DURATIONS

    shocks = []
    for _, r in df.iterrows():
        ac = str(r["asset_class"]).lower()
        t = str(r["ticker"]).upper()

        if "fixed" in ac or "bond" in ac or "fi" in ac:
            dur = float(dur_map.get(t, 6.0))
            shocks.append(-dur * dy)
        else:
            shocks.append(equity_shock)

    shocks = np.array(shocks)
    pnl = float((df["weight"].astype(float) * shocks).sum())

    return ScenarioResult(
        name="Rate Hike",
        shock_description=f"Rates +{rate_hike_bps:.0f}bps; FI shocked by duration; equities {equity_shock:.0%}",
        portfolio_pnl_pct=pnl,
    )


def scenario_rate_hike_dv01(
    positions: pd.DataFrame,
    rate_hike_bps: float = 100,
    portfolio_value: float = 1_000_000,
    equity_shock: float = -0.05,
    durations: dict | None = None,
) -> tuple[ScenarioResult, pd.DataFrame]:
    """DV01-based rate hike stress test.

    This is a more "risk desk" style stress test than the pure duration
    approximation.

    - For fixed income: DV01 ≈ Duration * MarketValue * 0.0001
    - P&L ≈ -DV01 * (rate_hike_bps)

    For equities, we keep a simple equity shock in return space.

    Returns (ScenarioResult, detail_df).
    """
    df = positions.copy()
    df["asset_class"] = df["asset_class"].fillna("equity")
    df["ticker"] = df["ticker"].astype(str).str.upper()

    dur_map = durations or DEFAULT_DURATIONS
    delta_bps = float(rate_hike_bps)

    details: list[ScenarioDetail] = []
    total_pnl = 0.0

    for _, r in df.iterrows():
        t = str(r["ticker"]).upper()
        ac = str(r["asset_class"]).lower()
        w = float(r["weight"])
        mv = w * float(portfolio_value)

        if "fixed" in ac or "bond" in ac or "fi" in ac:
            dur = float(dur_map.get(t, 6.0))
            dv01 = dur * mv * 0.0001
            pnl_value = -dv01 * delta_bps
        else:
            dv01 = 0.0
            pnl_value = mv * float(equity_shock)

        pnl_pct = pnl_value / float(portfolio_value)
        total_pnl += pnl_value

        details.append(
            ScenarioDetail(
                ticker=t,
                asset_class=str(r["asset_class"]),
                weight=w,
                dv01=dv01,
                pnl_value=pnl_value,
                pnl_pct=pnl_pct,
            )
        )

    detail_df = pd.DataFrame([d.__dict__ for d in details]).sort_values("pnl_value")

    res = ScenarioResult(
        name="Rate Hike (DV01)",
        shock_description=f"Rates +{rate_hike_bps:.0f}bps; FI shocked via DV01; equities {equity_shock:.0%}",
        portfolio_pnl_pct=float(total_pnl / float(portfolio_value)),
    )

    return res, detail_df


def scenario_factor_stress(
    returns_wide: pd.DataFrame,
    weights: pd.Series,
    factors: pd.DataFrame,
    equity_shock: float = -0.30,
    rate_hike_bps: float = 200,
) -> tuple[ScenarioResult, pd.DataFrame, pd.DataFrame]:
    """Factor-model stress scenario.

    Uses OLS betas to an equity factor + rate factor to translate macro
    shocks into expected asset returns.

    Returns (ScenarioResult, betas_df, detail_df).
    """
    if returns_wide.empty or factors.empty:
        return (
            ScenarioResult(
                name="Factor Stress",
                shock_description="Missing factor data (need SPY + DGS10)",
                portfolio_pnl_pct=float("nan"),
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    betas = estimate_betas_ols(returns_wide, factors)
    if betas.empty:
        return (
            ScenarioResult(
                name="Factor Stress",
                shock_description="Not enough overlapping data for beta estimation",
                portfolio_pnl_pct=float("nan"),
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    pnl, detail = factor_shock_pnl(
        weights=weights,
        betas=betas,
        equity_shock=float(equity_shock),
        rate_shock_bps=float(rate_hike_bps),
        include_alpha=False,
    )

    res = ScenarioResult(
        name="Factor Stress (Betas)",
        shock_description=f"Equity factor {equity_shock:.0%}, 10Y +{rate_hike_bps:.0f}bps",
        portfolio_pnl_pct=float(pnl),
    )
    return res, betas.reset_index(), detail

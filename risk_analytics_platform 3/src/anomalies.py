"""Simple anomaly detection for risk monitoring.

This is intentionally lightweight and interview-friendly:
- Flags return outliers using rolling z-score
- Flags volatility spikes when rolling vol exceeds a percentile

This mirrors common 'risk desk' monitoring heuristics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std(ddof=1)
    z = (series - mu) / sigma
    z.name = f"zscore_{window}d"
    return z


def detect_return_anomalies(
    port_returns: pd.Series,
    window: int = 60,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    z = rolling_zscore(port_returns, window=window)
    out = pd.DataFrame({
        "return": port_returns,
        "zscore": z,
        "is_anomaly": z.abs() >= z_thresh,
    })
    return out.dropna()


def detect_vol_spikes(
    rolling_vol: pd.Series,
    percentile: float = 0.95,
) -> pd.DataFrame:
    threshold = float(rolling_vol.quantile(percentile))
    out = pd.DataFrame({
        "rolling_vol": rolling_vol,
        "threshold": threshold,
        "is_spike": rolling_vol >= threshold,
    })
    return out.dropna()

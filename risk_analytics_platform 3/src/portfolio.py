"""Portfolio utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PortfolioDef:
    name: str
    positions: pd.DataFrame  # columns: ticker, asset_class, weight


def normalise_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weight"] = out["weight"].astype(float)
    total = out["weight"].sum()
    if total == 0:
        raise ValueError("Total portfolio weight is 0")
    out["weight"] = out["weight"] / total
    return out


def validate_weights(df: pd.DataFrame, tol: float = 1e-6) -> None:
    s = float(df["weight"].sum())
    if abs(s - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0, got {s:.6f}. Use normalise_weights() or adjust inputs.")


def positions_to_dicts(df: pd.DataFrame) -> list[dict]:
    return df[["ticker", "asset_class", "weight"]].to_dict(orient="records")

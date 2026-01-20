"""Helpers to pull data from SQL into pandas."""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select

from .db import MarketPrice, FredRate, get_session


def load_prices(
    tickers: Iterable[str],
    start: str | dt.date = "2016-01-01",
    end: Optional[str | dt.date] = None,
) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers if t]
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    if end is None:
        end = dt.date.today()

    sess = get_session()
    try:
        stmt = (
            select(MarketPrice.date, MarketPrice.ticker, MarketPrice.adj_close)
            .where(MarketPrice.ticker.in_(tickers))
            .where(MarketPrice.date >= start)
            .where(MarketPrice.date <= end)
        )
        rows = sess.execute(stmt).all()
    finally:
        sess.close()

    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_fred(
    series: Iterable[str],
    start: str | dt.date = "2016-01-01",
    end: Optional[str | dt.date] = None,
) -> pd.DataFrame:
    series = [s.upper().strip() for s in series if s]
    if not series:
        return pd.DataFrame(columns=["date", "series", "value"])

    if end is None:
        end = dt.date.today()

    sess = get_session()
    try:
        stmt = (
            select(FredRate.date, FredRate.series, FredRate.value)
            .where(FredRate.series.in_(series))
            .where(FredRate.date >= start)
            .where(FredRate.date <= end)
        )
        rows = sess.execute(stmt).all()
    finally:
        sess.close()

    df = pd.DataFrame(rows, columns=["date", "series", "value"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

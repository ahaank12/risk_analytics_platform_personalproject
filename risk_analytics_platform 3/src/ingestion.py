"""Market data ingestion.

Equities + fixed income instruments are ingested from Yahoo Finance (via yfinance).
Rate series are optionally ingested from FRED (via pandas-datareader).

Design goal:
- Store raw data in SQL (SQLite)
- Analytics pull data from SQL into pandas
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from sqlalchemy import select

from .db import MarketPrice, FredRate, get_session


def _normalize_price_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert yfinance OHLCV dataframe to standard columns."""
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]

    if "adj_close" not in out.columns:
        if "adjclose" in out.columns:
            out = out.rename(columns={"adjclose": "adj_close"})
        elif "adj_close" not in out.columns and "adj_close" in out.columns:
            pass

    if "adj_close" not in out.columns:
        out["adj_close"] = out.get("close")

    out = out.reset_index()
    if "date" not in out.columns and "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})

    out["ticker"] = ticker
    out["date"] = pd.to_datetime(out["date"]).dt.date

    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    for c in cols:
        if c not in out.columns:
            out[c] = None

    return out[cols]


def _upsert_market_price(session, row: pd.Series) -> None:
    existing = session.execute(
        select(MarketPrice).where(MarketPrice.date == row["date"]).where(MarketPrice.ticker == row["ticker"])
    ).scalar_one_or_none()

    if existing:
        existing.open = float(row["open"]) if pd.notna(row["open"]) else None
        existing.high = float(row["high"]) if pd.notna(row["high"]) else None
        existing.low = float(row["low"]) if pd.notna(row["low"]) else None
        existing.close = float(row["close"]) if pd.notna(row["close"]) else None
        existing.adj_close = float(row["adj_close"]) if pd.notna(row["adj_close"]) else None
        existing.volume = float(row["volume"]) if pd.notna(row["volume"]) else None
        existing.source = "yahoo"
    else:
        session.add(
            MarketPrice(
                date=row["date"],
                ticker=row["ticker"],
                open=float(row["open"]) if pd.notna(row["open"]) else None,
                high=float(row["high"]) if pd.notna(row["high"]) else None,
                low=float(row["low"]) if pd.notna(row["low"]) else None,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                adj_close=float(row["adj_close"]) if pd.notna(row["adj_close"]) else None,
                volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
                source="yahoo",
            )
        )


def ingest_yahoo_prices(
    tickers: Iterable[str],
    start: str | dt.date = "2016-01-01",
    end: Optional[str | dt.date] = None,
    interval: str = "1d",
) -> dict:
    """Download and store OHLCV + adjusted close for tickers."""
    if end is None:
        end = dt.date.today()

    tickers = list(dict.fromkeys([t.upper().strip() for t in tickers if t]))
    if not tickers:
        return {"inserted": 0, "tickers": []}

    session = get_session()
    inserted = 0

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                auto_adjust=False,
                progress=False,
                actions=False,
            )
            if df is None or df.empty:
                continue

            norm = _normalize_price_df(df, ticker)
            for _, r in norm.iterrows():
                _upsert_market_price(session, r)
                inserted += 1

            session.commit()

        except Exception as e:
            session.rollback()
            print(f"[WARN] Failed to ingest {ticker}: {e}")

    session.close()
    return {"inserted": inserted, "tickers": tickers}


def _upsert_fred_rate(session, date, series: str, value) -> None:
    existing = session.execute(
        select(FredRate).where(FredRate.date == date).where(FredRate.series == series)
    ).scalar_one_or_none()

    if existing:
        existing.value = value
        existing.source = "fred"
    else:
        session.add(FredRate(date=date, series=series, value=value, source="fred"))


def ingest_fred_series(
    series: Iterable[str],
    start: str | dt.date = "2016-01-01",
    end: Optional[str | dt.date] = None,
) -> dict:
    """Download and store FRED series (e.g., DGS10, DGS2)."""
    if end is None:
        end = dt.date.today()

    series = list(dict.fromkeys([s.upper().strip() for s in series if s]))
    if not series:
        return {"inserted": 0, "series": []}

    session = get_session()
    inserted = 0

    for code in series:
        try:
            df = pdr.DataReader(code, "fred", start=start, end=end)
            if df is None or df.empty:
                continue

            df = df.reset_index()
            date_col = "DATE" if "DATE" in df.columns else "Date" if "Date" in df.columns else "index"
            if date_col == "index":
                df = df.rename(columns={"index": "date"})
                date_col = "date"

            df["date"] = pd.to_datetime(df[date_col]).dt.date

            val_col = code if code in df.columns else [c for c in df.columns if c not in {"date", date_col}][0]
            df["value"] = pd.to_numeric(df[val_col], errors="coerce")

            for _, r in df.iterrows():
                value = float(r["value"]) if pd.notna(r["value"]) else None
                _upsert_fred_rate(session, r["date"], code, value)
                inserted += 1

            session.commit()

        except Exception as e:
            session.rollback()
            print(f"[WARN] Failed to ingest FRED {code}: {e}")

    session.close()
    return {"inserted": inserted, "series": series}

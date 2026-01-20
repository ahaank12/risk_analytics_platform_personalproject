"""
Ingest market data into SQLite.

- Downloads OHLCV from Yahoo Finance (yfinance)
- Writes into: market_prices table
- Handles both single-ticker and multi-ticker formats safely
- Prevents NULL close/adj_close inserts
"""

import os
import argparse
import sqlite3
import pandas as pd
import yfinance as yf

DEFAULT_DB = "data/risk.db"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (optional)")
    p.add_argument("--tickers", type=str, default="SPY,AAPL,MSFT,TLT,IEF,LQD",
                   help="Comma-separated tickers")
    p.add_argument("--db", type=str, default=None, help="SQLite DB path override (optional)")
    p.add_argument("--replace", action="store_true",
                   help="If set, delete existing rows for tickers before inserting")
    return p.parse_args()


def db_path_from_env_or_default(cli_db):
    if cli_db:
        return cli_db
    url = os.getenv("DATABASE_URL")
    if url and url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "")
    return DEFAULT_DB


def ensure_tables(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS market_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATE NOT NULL,
        ticker VARCHAR(32) NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        adj_close FLOAT,
        volume FLOAT,
        source VARCHAR(32) NOT NULL,
        UNIQUE(date, ticker, source)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date ON market_prices(ticker, date);")
    conn.commit()


def yf_download(tickers, start, end):
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance download returned empty data. Try again shortly.")
    return df


def normalise_to_long(df, tickers):
    records = []

    if isinstance(df.columns, pd.MultiIndex):
        # Most reliable: sub = df[ticker] (ticker as top-level) OR xs on level=1
        for t in tickers:
            if t in df.columns.get_level_values(0):
                sub = df[t].copy()
            elif ("Close", t) in df.columns:
                sub = df.xs(t, axis=1, level=1).copy()
            else:
                print(f"⚠️ Could not locate columns for {t}. Skipping.")
                continue

            sub = sub.reset_index()
            for _, r in sub.iterrows():
                records.append({
                    "date": pd.to_datetime(r["Date"]).date(),
                    "ticker": t,
                    "open": r.get("Open"),
                    "high": r.get("High"),
                    "low": r.get("Low"),
                    "close": r.get("Close"),
                    "adj_close": r.get("Adj Close"),
                    "volume": r.get("Volume"),
                })
    else:
        # Single ticker case
        t = tickers[0]
        sub = df.reset_index()
        for _, r in sub.iterrows():
            records.append({
                "date": pd.to_datetime(r["Date"]).date(),
                "ticker": t,
                "open": r.get("Open"),
                "high": r.get("High"),
                "low": r.get("Low"),
                "close": r.get("Close"),
                "adj_close": r.get("Adj Close"),
                "volume": r.get("Volume"),
            })

    out = pd.DataFrame.from_records(records)
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_prices(long_df):
    if long_df.empty:
        raise RuntimeError("No records produced after normalising Yahoo data.")

    total = len(long_df)
    null_close = long_df["close"].isna().sum()
    null_adj = long_df["adj_close"].isna().sum()

    if null_close / total > 0.02:
        raise RuntimeError(f"Too many NULL close values ({null_close}/{total}).")
    if null_adj / total > 0.02:
        raise RuntimeError(f"Too many NULL adj_close values ({null_adj}/{total}).")


def upsert_market_prices(conn, long_df, replace, tickers):
    ensure_tables(conn)

    if replace:
        q = f"DELETE FROM market_prices WHERE ticker IN ({','.join(['?']*len(tickers))})"
        conn.execute(q, tickers)
        conn.commit()

    long_df = long_df.copy()
    long_df["source"] = "yahoo"

    rows = []
    for r in long_df.itertuples(index=False):
        rows.append((
            str(r.date),
            r.ticker,
            None if pd.isna(r.open) else float(r.open),
            None if pd.isna(r.high) else float(r.high),
            None if pd.isna(r.low) else float(r.low),
            None if pd.isna(r.close) else float(r.close),
            None if pd.isna(r.adj_close) else float(r.adj_close),
            None if pd.isna(r.volume) else float(r.volume),
            r.source
        ))

    conn.executemany("""
    INSERT OR REPLACE INTO market_prices
    (date, ticker, open, high, low, close, adj_close, volume, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    db_path = db_path_from_env_or_default(args.db)

    print("✅ Ingestion starting")
    print("DB:", db_path)
    print("Tickers:", tickers)
    print("Start:", args.start, "End:", args.end if args.end else "(open-ended)")

    conn = sqlite3.connect(db_path)

    df = yf_download(tickers=tickers, start=args.start, end=args.end)
    long_df = normalise_to_long(df, tickers)
    validate_prices(long_df)
    upsert_market_prices(conn, long_df, replace=args.replace, tickers=tickers)

    total = conn.execute("SELECT COUNT(*) FROM market_prices").fetchone()[0]
    null_close = conn.execute("SELECT COUNT(*) FROM market_prices WHERE close IS NULL").fetchone()[0]

    print("\n✅ Ingestion complete")
    print("Total market_prices rows:", total)
    print("NULL close rows:", null_close)

    conn.close()


if __name__ == "__main__":
    main()

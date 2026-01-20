"""Compute and store a risk snapshot for one or more portfolios.

Usage examples:
    python scripts/snapshot_risk.py --portfolio "Demo Portfolio" --confidence 0.95
    python scripts/snapshot_risk.py --all --confidence 0.99

Snapshots are stored in the `risk_snapshots` table and can be visualised in the
Streamlit dashboard under the "History" tab.
"""

from __future__ import annotations

import argparse
import datetime as dt

from src.db import (
    get_session,
    list_portfolios,
    get_positions_df,
    upsert_risk_snapshot,
)
from src.utils import load_prices
from src.risk import compute_returns, pivot_returns, compute_all_metrics


def snapshot_one(portfolio_name: str, confidence: float, start: dt.date | None = None) -> None:
    alpha = 1.0 - confidence

    sess = get_session()
    try:
        pos = get_positions_df(sess, portfolio_name)
    finally:
        sess.close()

    if pos.empty:
        print(f"[WARN] No positions for {portfolio_name}")
        return

    tickers = pos["ticker"].tolist()
    prices = load_prices(tickers, start=start or dt.date(2016, 1, 1), end=dt.date.today())

    if prices.empty:
        print(f"[WARN] No prices in DB for {portfolio_name}. Run scripts/ingest_market_data.py")
        return

    ret_df = compute_returns(prices)
    ret_wide = pivot_returns(ret_df)
    if ret_wide.empty:
        print(f"[WARN] Not enough data overlap for {portfolio_name}")
        return

    as_of = ret_wide.index.max()
    weights = pos.set_index("ticker")["weight"]

    metrics = compute_all_metrics(ret_wide, weights, alpha=alpha, n_sims=10000)

    payload = {
        "ann_vol": float(metrics.vol_annualised),
        "var_historical": float(metrics.var_hist),
        "es_historical": float(metrics.es_hist),
        "var_parametric": float(metrics.var_parametric),
        "es_parametric": float(metrics.es_parametric),
        "var_monte_carlo": float(metrics.var_mc),
        "es_monte_carlo": float(metrics.es_mc),
    }

    sess = get_session()
    try:
        upsert_risk_snapshot(sess, portfolio_name, as_of, confidence, payload)
    finally:
        sess.close()

    print(f"✅ Stored snapshot: {portfolio_name} | {as_of} | {confidence:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    sess = get_session()
    try:
        names = list_portfolios(sess)
    finally:
        sess.close()

    if args.all:
        targets = names
    else:
        targets = [args.portfolio] if args.portfolio else [names[0] if names else "Demo Portfolio"]

    for p in targets:
        if p:
            snapshot_one(p, confidence=args.confidence)


if __name__ == "__main__":
    main()

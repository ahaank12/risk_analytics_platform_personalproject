"""Generate a PDF risk report from the CLI.

Usage:
  python scripts/generate_report.py --portfolio "Demo Portfolio" --start 2020-01-01 --confidence 0.95

The report is saved to ./reports/
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from src.db import get_session, get_positions_df
from src.utils import load_prices
from src.risk import compute_returns, pivot_returns, portfolio_returns, rolling_volatility, compute_all_metrics
from src.attribution import (
    component_var_parametric,
    component_var_historical,
    component_var_monte_carlo,
    es_contribution_historical,
)
from src.reporting import generate_risk_report_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default="Demo Portfolio")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n_sims", type=int, default=10000)
    args = parser.parse_args()

    alpha = 1.0 - float(args.confidence)

    sess = get_session()
    try:
        positions = get_positions_df(sess, args.portfolio)
    finally:
        sess.close()

    if positions.empty:
        raise SystemExit("Portfolio not found or empty. Seed data first.")

    tickers = positions["ticker"].tolist()
    prices = load_prices(tickers, start=args.start, end=args.end)
    if prices.empty:
        raise SystemExit("No price data found for tickers/date range. Ingest market data first.")

    ret_df = compute_returns(prices)
    ret_wide = pivot_returns(ret_df)
    weights = positions.set_index("ticker")["weight"]
    port_rets = portfolio_returns(ret_wide, weights)
    rolling_vol = rolling_volatility(port_rets)

    m = compute_all_metrics(ret_wide, weights, alpha=alpha, n_sims=int(args.n_sims))

    comp_hist = component_var_historical(ret_wide, weights, alpha=alpha)
    comp_para = component_var_parametric(ret_wide, weights, alpha=alpha)
    comp_mc = component_var_monte_carlo(ret_wide, weights, alpha=alpha, n_sims=int(args.n_sims))
    es_contrib = es_contribution_historical(ret_wide, weights, alpha=alpha)

    attrib_pack = {
        f"Component VaR (Historical) @ {args.confidence:.0%}": comp_hist,
        f"Component VaR (Monte Carlo) @ {args.confidence:.0%}": comp_mc,
        f"Component VaR (Parametric) @ {args.confidence:.0%}": comp_para,
        f"ES Contributions (Historical) @ {args.confidence:.0%}": es_contrib,
    }

    as_of = port_rets.index.max()
    pdf = generate_risk_report_pdf(
        portfolio_name=args.portfolio,
        as_of_date=as_of,
        positions=positions,
        port_rets=port_rets,
        rolling_vol=rolling_vol,
        metrics={
            "ann_vol": m.vol_annualised,
            "var_historical": m.var_hist,
            "es_historical": m.es_hist,
            "var_parametric": m.var_parametric,
            "es_parametric": m.es_parametric,
            "var_monte_carlo": m.var_mc,
            "es_monte_carlo": m.es_mc,
        },
        attribution_tables=attrib_pack,
    )

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"risk_report_{args.portfolio.replace(' ', '_').lower()}_{as_of}.pdf"
    out_path.write_bytes(pdf)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

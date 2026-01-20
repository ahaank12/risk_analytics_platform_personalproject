"""Create or update demo portfolios in the database.

This gives interviewers multiple choices and makes the dashboard feel more like
an internal tool.
"""

from __future__ import annotations

import pandas as pd

from src.db import get_session, upsert_portfolio, set_positions
from src.portfolio import normalise_weights, positions_to_dicts


DEMO_PORTFOLIOS = {
    "Demo Portfolio": [
        {"ticker": "SPY", "asset_class": "equity", "weight": 0.35},
        {"ticker": "AAPL", "asset_class": "equity", "weight": 0.15},
        {"ticker": "MSFT", "asset_class": "equity", "weight": 0.10},
        {"ticker": "TLT", "asset_class": "fixed_income", "weight": 0.15},
        {"ticker": "IEF", "asset_class": "fixed_income", "weight": 0.15},
        {"ticker": "LQD", "asset_class": "fixed_income", "weight": 0.10},
    ],
    "60/40 Balanced": [
        {"ticker": "SPY", "asset_class": "equity", "weight": 0.60},
        {"ticker": "IEF", "asset_class": "fixed_income", "weight": 0.25},
        {"ticker": "LQD", "asset_class": "fixed_income", "weight": 0.15},
    ],
    "Equity Tilt": [
        {"ticker": "SPY", "asset_class": "equity", "weight": 0.55},
        {"ticker": "AAPL", "asset_class": "equity", "weight": 0.20},
        {"ticker": "MSFT", "asset_class": "equity", "weight": 0.20},
        {"ticker": "IEF", "asset_class": "fixed_income", "weight": 0.05},
    ],
}


def main() -> None:
    sess = get_session()
    try:
        for name, rows in DEMO_PORTFOLIOS.items():
            positions = pd.DataFrame(rows)
            positions = normalise_weights(positions)

            portfolio = upsert_portfolio(sess, name)
            set_positions(sess, portfolio, positions_to_dicts(positions))

            print(f"✅ Seeded portfolio: {name}")
            print(positions)
    finally:
        sess.close()


if __name__ == "__main__":
    main()

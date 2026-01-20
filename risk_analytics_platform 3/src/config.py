"""Project configuration.

You can override most settings using environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Settings:
    # SQLite database file path
    db_path: Path = Path(os.getenv("RISK_DB_PATH", str(DATA_DIR / "risk.db")))

    # Default portfolio universe (equities + fixed income ETFs)
    default_tickers: tuple[str, ...] = (
        "SPY",  # US equities
        "AAPL",  # single-name equity
        "MSFT",
        "TLT",  # long duration Treasuries ETF
        "IEF",  # intermediate Treasuries ETF
        "LQD",  # investment grade credit ETF
    )

    # FRED series for rates (optional)
    fred_series: tuple[str, ...] = (
        "DGS10",  # 10Y Treasury yield
        "DGS2",   # 2Y Treasury yield
    )


SETTINGS = Settings()

"""Initialise the SQLite database.

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --reset

`--reset` deletes the existing DB file (useful when upgrading schemas).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import SETTINGS
from src.db import init_db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete existing DB file before creating tables")
    args = parser.parse_args()

    db_path = Path(SETTINGS.db_path)

    if args.reset and db_path.exists():
        db_path.unlink()
        print(f"🧹 Deleted existing DB: {db_path}")

    init_db()
    print(f"✅ Database initialised: {db_path}")


if __name__ == "__main__":
    main()

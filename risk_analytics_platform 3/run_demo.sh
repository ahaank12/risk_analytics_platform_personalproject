#!/usr/bin/env bash
set -euo pipefail

# One-click local demo runner (macOS/Linux)
# - creates venv
# - installs requirements
# - resets & seeds SQLite DB
# - launches Streamlit app

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Please install Python 3.10+ and rerun." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/init_db.py --reset
python scripts/ingest_market_data.py --start 2018-01-01
python scripts/seed_portfolio.py

echo "
✅ Demo is ready. Launching Streamlit app..." 
echo "Tip: If this is your first run, data ingestion may take a couple of minutes." 

streamlit run app.py

# One-click local demo runner (Windows PowerShell)
# - creates venv
# - installs requirements
# - resets & seeds SQLite DB
# - launches Streamlit app

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

# Pick python executable
$python = "python"
if (Get-Command python3 -ErrorAction SilentlyContinue) {
  $python = "python3"
}

Write-Host "Using Python: $python"

# Create venv if missing
if (-not (Test-Path ".venv")) {
  & $python -m venv .venv
}

# Activate venv
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts\init_db.py --reset
python scripts\ingest_market_data.py --start 2018-01-01
python scripts\seed_portfolio.py

Write-Host "`n✅ Demo is ready. Launching Streamlit app..."
Write-Host "Tip: If this is your first run, data ingestion may take a couple of minutes."

streamlit run app.py

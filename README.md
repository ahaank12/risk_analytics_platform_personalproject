# End-to-End Risk Analytics Platform (Python + SQL + Streamlit)

A compact risk analytics system focused on **equities + fixed income**, built as an end-to-end portfolio risk demo.

✅ Pulls market data (Yahoo Finance)  
✅ Stores data + portfolio definitions in **SQL (SQLite)**  
✅ Computes risk metrics (Volatility, VaR, Expected Shortfall)  
✅ Flags anomalies (return outliers, volatility spikes)  
✅ Runs stress tests (2008-style crash, rate hikes via **Duration + DV01**)  
✅ Interactive **Streamlit** dashboard
✅ Please navigate to the **results** folder to take a look at an **example risk report** and **screenshots of the dashboard**

✅ **Multi-portfolio** support
✅ **Risk attribution** (Component VaR, ES contributions, vol contributions)
✅ **Daily risk snapshots** saved to SQL for history charts



## 🚀 One-click demo
Run the full end-to-end workflow (create venv → ingest data → seed portfolios → launch Streamlit) with one command:

**macOS/Linux**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File ./run_demo.ps1
```

### Extra features
- **Historical + Monte Carlo Component VaR** (scenario-based)
- **Factor-model stress framework** (SPY equity factor + 10Y yield changes)
- **Client-style PDF export** (downloadable report from the dashboard)

---

## 📸 Dashboard

![Demo Flow](docs/screenshots/demo_flow.gif)

A quick walkthrough of the app tabs and outputs.

![Overview](docs/screenshots/overview.png)

### Tabs
- **Overview**: portfolio performance, distribution, anomalies, save snapshots
- **Attribution**: Component VaR, ES contributions, volatility contributions
- **Stress**: 2008 crash + rate hike stress (duration + DV01) + **factor stress (betas)**
- **History**: risk snapshots over time (stored in SQLite)
- **Rates**: FRED series for macro context (optional)
- **Report**: generate + download a PDF risk report

![Attribution](docs/screenshots/attribution.png)
![Stress](docs/screenshots/stress.png)
![History](docs/screenshots/history.png)

---

## 🧱 Tech Stack
- **Python**: pandas, numpy, scipy
- **Data APIs**: yfinance (Yahoo Finance), pandas-datareader (FRED)
- **SQL**: SQLite + SQLAlchemy
- **Dashboard**: Streamlit + Plotly
- **Reporting**: reportlab + matplotlib (PDF export)

---

## ⚙️ Local Setup

### 1) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Initialize database
```bash
python scripts/init_db.py --reset
```

### 3) Ingest market data
Default universe includes equities + bond ETFs:
- SPY, AAPL, MSFT
- TLT, IEF, LQD

```bash
python scripts/ingest_market_data.py --start 2018-01-01
```

### 4) Seed a demo portfolio
```bash
python scripts/seed_portfolio.py
```

### 4b) (Optional) Store risk snapshots in SQL
```bash
python scripts/snapshot_risk.py --all --confidence 0.95
```

### 5) Run the dashboard
```bash
streamlit run app.py
```

---

## 🗄️ Why SQL is used here
This project intentionally uses SQL for:
- **Raw market data** persistence (prices, rates)
- **Portfolio definitions** (weights, asset classes)

Risk metrics are computed in-memory using pandas for clarity and transparency.

---

## 📁 Project Structure
```
risk_analytics_platform/
  app.py
  requirements.txt
  README.md
  data/
    risk.db
  scripts/
    init_db.py
    ingest_market_data.py
    seed_portfolio.py
    snapshot_risk.py
  src/
    config.py
    db.py
    ingestion.py
    utils.py
    portfolio.py
    risk.py
    scenarios.py
    anomalies.py
    attribution.py
  docs/
    screenshots/
      overview.png
      attribution.png
      stress.png
      history.png
  tests/
    test_risk.py
```

---

## What makes this project useful
- **End-to-end pipeline**: ingestion -> SQL -> analytics -> dashboard
- **Multiple VaR methods**: Historical, Parametric, Monte Carlo
- **Explanation layer**: attribution + scenario stress results
- **Good engineering habits**: modular src/ folder + scripts + tests

---

## License
MIT

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import SETTINGS
from src.db import (
    get_session,
    list_portfolios,
    get_positions_df,
    upsert_portfolio,
    set_positions,
    get_risk_snapshots_df,
    upsert_risk_snapshot,
)
from src.portfolio import normalise_weights, validate_weights, positions_to_dicts
from src.utils import load_prices, load_fred
from src.risk import compute_returns, pivot_returns, portfolio_returns, rolling_volatility, compute_all_metrics
from src.anomalies import detect_return_anomalies, detect_vol_spikes
from src.scenarios import scenario_equity_crash, scenario_rate_hike, scenario_rate_hike_dv01
from src.scenarios import scenario_factor_stress
from src.attribution import (
    component_var_parametric,
    component_var_historical,
    component_var_monte_carlo,
    volatility_contribution,
    es_contribution_historical,
)
from src.factor_model import prepare_factors
from src.reporting import generate_risk_report_pdf


st.set_page_config(page_title="Risk Analytics Platform", layout="wide")

st.title("End-to-End Risk Analytics Platform")
st.caption("Fixed Income + Equities • Python + SQL (SQLite) • Streamlit Dashboard")

with st.expander("What this demo does", expanded=False):
    st.markdown(
        """
This is a mini **risk desk** system:

- Ingests market data (Yahoo Finance) into **SQLite**
- Builds a portfolio (weights in SQL)
- Computes portfolio risk metrics (VaR, Expected Shortfall, volatility)
- Flags anomalies (return outliers, vol spikes)
- Runs stress tests (2008-style crash, rate hike)

If you haven't ingested data yet, run:

```bash
python scripts/init_db.py
python scripts/ingest_market_data.py
python scripts/seed_portfolio.py
streamlit run app.py
```
"""
    )

# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Controls")

    sess = get_session()
    try:
        portfolios = list_portfolios(sess)
    finally:
        sess.close()

    if not portfolios:
        st.warning("No portfolio found in DB. Run scripts/seed_portfolio.py")
        portfolio_name = "Demo Portfolio"
    else:
        portfolio_name = st.selectbox("Portfolio", portfolios, index=0)

    st.divider()
    st.subheader("Portfolio Management")
    new_name = st.text_input("Create new portfolio", value="")
    if st.button("Create", use_container_width=True, disabled=(len(new_name.strip()) == 0)):
        sess = get_session()
        try:
            upsert_portfolio(sess, new_name.strip())
            st.success(f"Created: {new_name.strip()}")
        finally:
            sess.close()

    colA, colB = st.columns(2)
    with colA:
        start_date = st.date_input("Start", value=dt.date(2020, 1, 1))
    with colB:
        end_date = st.date_input("End", value=dt.date.today())

    confidence = st.slider("Confidence level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    alpha = 1.0 - confidence

    n_sims = st.slider("Monte Carlo sims", min_value=2000, max_value=30000, value=10000, step=1000)

    st.divider()
    st.subheader("Stress Tests")
    equity_crash = st.slider("Equity crash shock", min_value=-0.70, max_value=-0.05, value=-0.30, step=0.05)
    fi_crash = st.slider("FI shock", min_value=-0.40, max_value=0.10, value=-0.10, step=0.05)
    rate_hike = st.slider("Rate hike (bps)", min_value=25, max_value=400, value=100, step=25)
    portfolio_value = st.number_input("Portfolio value (for DV01)", min_value=100000, max_value=100000000, value=1000000, step=100000)

# ----------------------------- Load portfolio -----------------------------
sess = get_session()
try:
    positions_df = get_positions_df(sess, portfolio_name)
finally:
    sess.close()

if positions_df.empty:
    st.error("No positions loaded. Seed a portfolio first (scripts/seed_portfolio.py).")
    st.stop()

st.subheader("Portfolio")
st.write("Edit weights below (they will be normalised for analytics).")
edited = st.data_editor(
    positions_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=["ticker", "asset_class"],
    key="positions_editor",
)

save_col1, save_col2 = st.columns([1, 3])
with save_col1:
    save_to_db = st.button("Save weights to DB")
with save_col2:
    st.caption("Saving persists weights for this portfolio in SQLite.")

try:
    edited = normalise_weights(edited)
    validate_weights(edited)
except Exception as e:
    st.error(f"Portfolio weights issue: {e}")
    st.stop()

weights = edited.set_index("ticker")["weight"]

if save_to_db:
    sess = get_session()
    try:
        p = upsert_portfolio(sess, portfolio_name)
        set_positions(sess, p, positions_to_dicts(edited))
        st.success("Saved weights to DB ✅")
    finally:
        sess.close()

# ----------------------------- Load market data -----------------------------
prices = load_prices(edited["ticker"].tolist(), start=start_date, end=end_date)

if prices.empty:
    st.error(
        "No market data found in DB for selected tickers/date range. "
        "Run scripts/ingest_market_data.py to populate SQLite."
    )
    st.stop()

ret_df = compute_returns(prices)
ret_wide = pivot_returns(ret_df)

if ret_wide.empty:
    st.error("Not enough overlapping data to compute returns. Try a different date range.")
    st.stop()

port_rets = portfolio_returns(ret_wide, weights)
rolling_vol = rolling_volatility(port_rets, window=30)

# ----------------------------- Metrics -----------------------------
metrics = compute_all_metrics(ret_wide, weights, alpha=alpha, n_sims=n_sims)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Ann. Volatility", f"{metrics.vol_annualised:.2%}")
kpi2.metric(f"VaR (Hist) @ {confidence:.0%}", f"{metrics.var_hist:.2%}")
kpi3.metric(f"ES (Hist) @ {confidence:.0%}", f"{metrics.es_hist:.2%}")
kpi4.metric(f"VaR (MC) @ {confidence:.0%}", f"{metrics.var_mc:.2%}")

tabs = st.tabs(["Overview", "Attribution", "Stress", "History", "Rates", "Report"])

with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Portfolio Performance")
        equity_curve = (1.0 + port_rets).cumprod()
        fig = px.line(
            equity_curve.reset_index(),
            x="date",
            y="portfolio_return",
            labels={"portfolio_return": "Cumulative Value"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Rolling Volatility (30d)")
        fig2 = px.line(
            rolling_vol.dropna().reset_index(),
            x="date",
            y=rolling_vol.name,
            labels={rolling_vol.name: "Ann. Vol"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Return Distribution")
    fig3 = px.histogram(port_rets, nbins=60, labels={"value": "Daily Return"})
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------- Anomalies -----------------------------
    st.subheader("Anomaly Monitoring")

    anom = detect_return_anomalies(port_rets, window=60, z_thresh=3.0)
    spikes = detect_vol_spikes(rolling_vol.dropna(), percentile=0.95)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Return outliers (z-score >= 3)**")
        outliers = anom[anom["is_anomaly"]].tail(15)
        st.dataframe(outliers, use_container_width=True)

    with col2:
        st.write("**Volatility spikes (top 5%)**")
        spikes_df = spikes[spikes["is_spike"]].tail(15)
        st.dataframe(spikes_df, use_container_width=True)

    st.divider()
    st.subheader("Save risk snapshot")
    st.caption("Stores current metrics into SQLite for history charts.")
    if st.button("Save snapshot", use_container_width=True):
        as_of = port_rets.index.max()
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
            st.success(f"Snapshot saved for {as_of} ✅")
        finally:
            sess.close()

with tabs[1]:
    st.subheader("Risk Attribution")
    st.caption("Component VaR and tail-risk contributions help explain where portfolio risk comes from.")

    comp_var_para = component_var_parametric(ret_wide, weights, alpha=alpha)
    comp_var_hist = component_var_historical(ret_wide, weights, alpha=alpha)
    comp_var_mc = component_var_monte_carlo(ret_wide, weights, alpha=alpha, n_sims=n_sims)
    es_contrib = es_contribution_historical(ret_wide, weights, alpha=alpha)
    vol_contrib = volatility_contribution(ret_wide, weights)

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Component VaR (Historical) @ {confidence:.0%}**")
        if not comp_var_hist.empty:
            fig = px.bar(comp_var_hist, x="ticker", y="component_var")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comp_var_hist, use_container_width=True)
        else:
            st.info("Not enough data to compute historical component VaR.")

    with c2:
        st.write(f"**Component VaR (Monte Carlo) @ {confidence:.0%}**")
        if not comp_var_mc.empty:
            fig = px.bar(comp_var_mc, x="ticker", y="component_var")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comp_var_mc, use_container_width=True)
        else:
            st.info("Not enough data to compute MC component VaR.")

    st.write(f"**Component VaR (Parametric) @ {confidence:.0%}**")
    if not comp_var_para.empty:
        fig = px.bar(comp_var_para, x="ticker", y="component_var")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp_var_para, use_container_width=True)
    else:
        st.info("Not enough data to compute parametric component VaR.")

    st.write(f"**ES Contributions (Historical) @ {confidence:.0%}**")
    if not es_contrib.empty:
        fig = px.bar(es_contrib, x="ticker", y="es_contribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(es_contrib, use_container_width=True)
    else:
        st.info("Not enough tail observations to compute ES contributions.")

    st.write("**Volatility Contributions (daily)**")
    if not vol_contrib.empty:
        fig = px.bar(vol_contrib, x="ticker", y="vol_contribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vol_contrib, use_container_width=True)

with tabs[2]:
    st.subheader("Stress Testing")

    crash_res = scenario_equity_crash(edited, equity_shock=equity_crash, fi_shock=fi_crash)
    rate_res = scenario_rate_hike(edited, rate_hike_bps=rate_hike)
    dv01_res, dv01_detail = scenario_rate_hike_dv01(
        edited,
        rate_hike_bps=rate_hike,
        portfolio_value=float(portfolio_value),
    )

    # Factor-model stress (requires SPY + DGS10 in DB)
    spy_prices = load_prices(["SPY"], start=start_date, end=end_date)
    spy_ret = pd.Series(dtype=float)
    if not spy_prices.empty:
        spy_ret_df = compute_returns(spy_prices)
        spy_wide = pivot_returns(spy_ret_df)
        if "SPY" in spy_wide.columns:
            spy_ret = spy_wide["SPY"]

    fred_df = load_fred(["DGS10"], start=start_date, end=end_date)
    rate_series = pd.Series(dtype=float)
    if not fred_df.empty:
        piv = fred_df.pivot(index="date", columns="series", values="value")
        if "DGS10" in piv.columns:
            rate_series = piv["DGS10"].dropna()

    factors = prepare_factors(spy_ret, rate_series)
    fac_res, fac_betas, fac_detail = scenario_factor_stress(
        returns_wide=ret_wide,
        weights=weights,
        factors=factors,
        equity_shock=equity_crash,
        rate_hike_bps=rate_hike,
    )

    stress_df = pd.DataFrame([
        {
            "Scenario": crash_res.name,
            "Shock": crash_res.shock_description,
            "Portfolio P&L": crash_res.portfolio_pnl_pct,
        },
        {
            "Scenario": rate_res.name,
            "Shock": rate_res.shock_description,
            "Portfolio P&L": rate_res.portfolio_pnl_pct,
        },
        {
            "Scenario": dv01_res.name,
            "Shock": dv01_res.shock_description,
            "Portfolio P&L": dv01_res.portfolio_pnl_pct,
        },
        {
            "Scenario": fac_res.name,
            "Shock": fac_res.shock_description,
            "Portfolio P&L": fac_res.portfolio_pnl_pct,
        },
    ])

    st.dataframe(
        stress_df.assign(**{"Portfolio P&L": stress_df["Portfolio P&L"].map(lambda x: f"{x:.2%}")}),
        use_container_width=True,
    )

    fig4 = px.bar(stress_df, x="Scenario", y="Portfolio P&L")
    st.plotly_chart(fig4, use_container_width=True)

    st.write("**DV01 Detail (Fixed Income sensitivity)**")
    if not dv01_detail.empty:
        total_dv01 = dv01_detail["dv01"].sum()
        st.metric("Total DV01", f"{total_dv01:,.0f} per 1bp")
        st.dataframe(dv01_detail, use_container_width=True)

    st.write("**Factor-model detail (beta-based, optional)**")
    if fac_detail is not None and not fac_detail.empty:
        st.dataframe(fac_detail, use_container_width=True)
        with st.expander("View estimated betas"):
            st.dataframe(fac_betas, use_container_width=True)
    else:
        st.info("Factor stress requires SPY + DGS10 data in the DB (ingest_market_data.py).")

with tabs[3]:
    st.subheader("Risk History")
    st.caption("Snapshots are stored in SQLite. Use the button in Overview or run scripts/snapshot_risk.py")

    sess = get_session()
    try:
        hist = get_risk_snapshots_df(sess, portfolio_name, confidence=confidence)
    finally:
        sess.close()

    if hist.empty:
        st.info("No snapshots found yet. Save one from the Overview tab.")
    else:
        hist = hist.sort_values("as_of_date")
        fig = px.line(hist, x="as_of_date", y=["ann_vol", "var_historical", "es_historical", "var_monte_carlo"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hist, use_container_width=True)

with tabs[4]:
    st.subheader("Rates (FRED, optional)")
    rate_df = load_fred(SETTINGS.fred_series, start=start_date, end=end_date)
    if rate_df.empty:
        st.info("No FRED rates found in DB. Run scripts/ingest_market_data.py to ingest rates series.")
    else:
        piv = rate_df.pivot(index="date", columns="series", values="value").dropna(how="all")
        fig5 = px.line(piv.reset_index(), x="date", y=piv.columns.tolist())
        st.plotly_chart(fig5, use_container_width=True)

with tabs[5]:
    st.subheader("PDF Risk Report")
    st.caption("Generate a client-style PDF report for sharing or interview demos.")

    report_as_of = port_rets.index.max()

    # Build attribution pack (top 12 rows each)
    attrib_pack = {
        f"Component VaR (Historical) @ {confidence:.0%}": comp_var_hist,
        f"Component VaR (Monte Carlo) @ {confidence:.0%}": comp_var_mc,
        f"Component VaR (Parametric) @ {confidence:.0%}": comp_var_para,
        f"ES Contributions (Historical) @ {confidence:.0%}": es_contrib,
    }

    stress_out = stress_df.copy()
    stress_out["Portfolio P&L"] = stress_out["Portfolio P&L"].astype(float)

    if st.button("Generate report", use_container_width=True):
        st.session_state["report_bytes"] = generate_risk_report_pdf(
            portfolio_name=portfolio_name,
            as_of_date=report_as_of,
            positions=edited,
            port_rets=port_rets,
            rolling_vol=rolling_vol,
            metrics={
                "ann_vol": metrics.vol_annualised,
                "var_historical": metrics.var_hist,
                "es_historical": metrics.es_hist,
                "var_parametric": metrics.var_parametric,
                "es_parametric": metrics.es_parametric,
                "var_monte_carlo": metrics.var_mc,
                "es_monte_carlo": metrics.es_mc,
            },
            attribution_tables=attrib_pack,
            stress_table=stress_out,
            factor_betas=fac_betas if "fac_betas" in locals() else None,
        )
        st.success("Report generated ✅")

    pdf_bytes = st.session_state.get("report_bytes")
    if pdf_bytes:
        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name=f"risk_report_{portfolio_name.replace(' ', '_').lower()}_{report_as_of}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Click **Generate report** to create a PDF.")

st.caption("Built to be readable, modular, and interview-friendly.")

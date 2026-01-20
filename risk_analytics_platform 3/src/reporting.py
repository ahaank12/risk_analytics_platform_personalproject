"""PDF reporting utilities.

This module generates a client-style risk report as a PDF.

Design goals:
 - Fast: works locally, no external dependencies beyond reportlab/matplotlib
 - Clear: summary KPIs, portfolio holdings, key charts, attribution + stress tests

The Streamlit app exposes this via a "Download PDF report" button.
"""

from __future__ import annotations

import io
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _table_from_df(df: pd.DataFrame, max_rows: int = 12) -> Table:
    if df is None or df.empty:
        data = [["No data"]]
        t = Table(data)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        return t

    d = df.copy().head(max_rows)
    # stringify with reasonable formatting
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")

    data = [d.columns.tolist()] + d.values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return t


def generate_risk_report_pdf(
    *,
    portfolio_name: str,
    as_of_date: dt.date,
    positions: pd.DataFrame,
    port_rets: pd.Series,
    rolling_vol: pd.Series,
    metrics: dict,
    attribution_tables: dict[str, pd.DataFrame] | None = None,
    stress_table: pd.DataFrame | None = None,
    factor_betas: pd.DataFrame | None = None,
) -> bytes:
    """Generate a PDF report and return raw PDF bytes."""

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = styles["BodyText"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Risk Report - {portfolio_name}")

    story = []
    story.append(Paragraph(f"Risk Report: {portfolio_name}", title_style))
    story.append(Paragraph(f"As of: {as_of_date}", body_style))
    story.append(Spacer(1, 10))

    # KPI table
    kpi_df = pd.DataFrame(
        [
            {
                "Metric": "Ann. Volatility",
                "Value": f"{float(metrics.get('ann_vol', np.nan)):.2%}",
            },
            {
                "Metric": "VaR (Hist)",
                "Value": f"{float(metrics.get('var_historical', np.nan)):.2%}",
            },
            {
                "Metric": "ES (Hist)",
                "Value": f"{float(metrics.get('es_historical', np.nan)):.2%}",
            },
            {
                "Metric": "VaR (Parametric)",
                "Value": f"{float(metrics.get('var_parametric', np.nan)):.2%}",
            },
            {
                "Metric": "VaR (Monte Carlo)",
                "Value": f"{float(metrics.get('var_monte_carlo', np.nan)):.2%}",
            },
        ]
    )
    story.append(Paragraph("Summary KPIs", styles["Heading2"]))
    story.append(_table_from_df(kpi_df, max_rows=10))
    story.append(Spacer(1, 8))

    # Positions
    story.append(Paragraph("Portfolio Holdings", styles["Heading2"]))
    story.append(_table_from_df(positions[["ticker", "asset_class", "weight"]], max_rows=20))
    story.append(Spacer(1, 8))

    # Charts
    story.append(Paragraph("Key Charts", styles["Heading2"]))

    # Equity curve
    eq_curve = (1.0 + port_rets).cumprod()
    fig1 = plt.figure(figsize=(6.5, 2.5))
    plt.plot(eq_curve.index, eq_curve.values)
    plt.title("Portfolio cumulative value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    story.append(Image(io.BytesIO(_fig_to_png_bytes(fig1)), width=500, height=190))
    story.append(Spacer(1, 6))

    # Rolling vol
    fig2 = plt.figure(figsize=(6.5, 2.5))
    rv = rolling_vol.dropna()
    if not rv.empty:
        plt.plot(rv.index, rv.values)
    plt.title("Rolling volatility (annualised)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    story.append(Image(io.BytesIO(_fig_to_png_bytes(fig2)), width=500, height=190))
    story.append(Spacer(1, 6))

    # Return histogram
    fig3 = plt.figure(figsize=(6.5, 2.5))
    plt.hist(port_rets.values, bins=60)
    plt.title("Daily return distribution")
    plt.tight_layout()
    story.append(Image(io.BytesIO(_fig_to_png_bytes(fig3)), width=500, height=190))
    story.append(Spacer(1, 10))

    # Attribution
    if attribution_tables:
        story.append(Paragraph("Risk Attribution", styles["Heading2"]))
        for name, df in attribution_tables.items():
            story.append(Paragraph(str(name), styles["Heading3"]))
            story.append(_table_from_df(df, max_rows=12))
            story.append(Spacer(1, 6))

    # Stress
    if stress_table is not None and not stress_table.empty:
        story.append(Paragraph("Stress Testing", styles["Heading2"]))
        story.append(_table_from_df(stress_table, max_rows=12))
        story.append(Spacer(1, 6))

    # Factor betas
    if factor_betas is not None and not factor_betas.empty:
        story.append(Paragraph("Factor Betas (Equity + Rates)", styles["Heading2"]))
        story.append(_table_from_df(factor_betas.reset_index(), max_rows=12))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Generated locally by Risk Analytics Platform (Python + SQL + Streamlit).",
            body_style,
        )
    )

    doc.build(story)
    return buf.getvalue()

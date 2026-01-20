"""Database layer (SQLite via SQLAlchemy).

This project uses SQL for:
- Persisting raw time-series data (prices, rates)
- Persisting portfolio definitions (positions + weights)

Risk analytics run in pandas for transparency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import SETTINGS


Base = declarative_base()


class MarketPrice(Base):
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)

    source = Column(String(32), nullable=False, default="yahoo")

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_market_prices_date_ticker"),
    )


class FredRate(Base):
    __tablename__ = "fred_rates"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    series = Column(String(32), nullable=False, index=True)
    value = Column(Float, nullable=True)
    source = Column(String(32), nullable=False, default="fred")

    __table_args__ = (
        UniqueConstraint("date", "series", name="uq_fred_rates_date_series"),
    )


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False, unique=True)

    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)

    ticker = Column(String(32), nullable=False)
    asset_class = Column(String(32), nullable=False, default="equity")
    weight = Column(Float, nullable=False)

    portfolio = relationship("Portfolio", back_populates="positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "ticker", name="uq_positions_portfolio_ticker"),
    )


class RiskResult(Base):
    __tablename__ = "risk_results"

    id = Column(Integer, primary_key=True)
    portfolio_name = Column(String(64), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)

    confidence = Column(Float, nullable=False)
    var_historical = Column(Float)
    var_parametric = Column(Float)
    var_monte_carlo = Column(Float)
    es_historical = Column(Float)
    ann_vol = Column(Float)

    __table_args__ = (
        UniqueConstraint("portfolio_name", "as_of_date", "confidence", name="uq_risk_results_key"),
    )


class RiskSnapshot(Base):
    """Daily (or ad-hoc) risk snapshot table.

    We keep this table separate from RiskResult to avoid breaking earlier
    iterations while adding richer analytics for recruiter-friendly history
    charts and attribution.
    """

    __tablename__ = "risk_snapshots"

    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    as_of_date = Column(Date, nullable=False, index=True)

    confidence = Column(Float, nullable=False)
    ann_vol = Column(Float)

    var_historical = Column(Float)
    es_historical = Column(Float)
    var_parametric = Column(Float)
    es_parametric = Column(Float)
    var_monte_carlo = Column(Float)
    es_monte_carlo = Column(Float)

    __table_args__ = (
        UniqueConstraint("portfolio_id", "as_of_date", "confidence", name="uq_risk_snapshots_key"),
    )


# --- Engine + session factory (singleton) ---
_ENGINE = None
_SessionFactory = None


def make_engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", future=True)


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = make_engine(Path(SETTINGS.db_path))
    return _ENGINE


def get_session():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
    return _SessionFactory()


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(get_engine())


# --- Portfolio helpers ---

def upsert_portfolio(session, name: str) -> Portfolio:
    existing = session.execute(select(Portfolio).where(Portfolio.name == name)).scalar_one_or_none()
    if existing:
        return existing
    p = Portfolio(name=name)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


def set_positions(session, portfolio: Portfolio, positions: Iterable[dict]) -> None:
    """Replace portfolio positions.

    positions: iterable of dicts with keys: ticker, asset_class, weight
    """
    portfolio.positions = []
    session.flush()

    for row in positions:
        pos = Position(
            portfolio_id=portfolio.id,
            ticker=str(row["ticker"]).upper(),
            asset_class=str(row.get("asset_class", "equity")),
            weight=float(row["weight"]),
        )
        session.add(pos)

    session.commit()


def get_positions(session, portfolio_name: str) -> list[Position]:
    p = session.execute(select(Portfolio).where(Portfolio.name == portfolio_name)).scalar_one_or_none()
    if not p:
        return []
    return list(p.positions)


def list_portfolios(session) -> list[str]:
    rows = session.execute(select(Portfolio.name)).all()
    return [r[0] for r in rows]


def get_portfolio_by_name(session, portfolio_name: str) -> Portfolio | None:
    return session.execute(select(Portfolio).where(Portfolio.name == portfolio_name)).scalar_one_or_none()


def upsert_risk_snapshot(
    session,
    portfolio_name: str,
    as_of_date,
    confidence: float,
    metrics: dict,
) -> None:
    """Insert or update a risk snapshot."""

    p = get_portfolio_by_name(session, portfolio_name)
    if not p:
        return

    existing = session.execute(
        select(RiskSnapshot)
        .where(RiskSnapshot.portfolio_id == p.id)
        .where(RiskSnapshot.as_of_date == as_of_date)
        .where(RiskSnapshot.confidence == confidence)
    ).scalar_one_or_none()

    if existing:
        for k, v in metrics.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
    else:
        session.add(
            RiskSnapshot(
                portfolio_id=p.id,
                as_of_date=as_of_date,
                confidence=confidence,
                **metrics,
            )
        )

    session.commit()


def get_risk_snapshots_df(session, portfolio_name: str, confidence: float):
    """Return snapshot history as a pandas DataFrame."""
    import pandas as pd

    p = get_portfolio_by_name(session, portfolio_name)
    if not p:
        return pd.DataFrame()

    rows = session.execute(
        select(RiskSnapshot)
        .where(RiskSnapshot.portfolio_id == p.id)
        .where(RiskSnapshot.confidence == confidence)
        .order_by(RiskSnapshot.as_of_date.asc())
    ).scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "as_of_date": r.as_of_date,
                "confidence": r.confidence,
                "ann_vol": r.ann_vol,
                "var_historical": r.var_historical,
                "es_historical": r.es_historical,
                "var_parametric": r.var_parametric,
                "es_parametric": r.es_parametric,
                "var_monte_carlo": r.var_monte_carlo,
                "es_monte_carlo": r.es_monte_carlo,
            }
            for r in rows
        ]
    )


def get_positions_df(session, portfolio_name: str):
    """Return positions as a pandas DataFrame."""
    import pandas as pd

    pos = get_positions(session, portfolio_name)
    if not pos:
        return pd.DataFrame(columns=["ticker", "asset_class", "weight"])

    return pd.DataFrame([
        {"ticker": p.ticker, "asset_class": p.asset_class, "weight": p.weight}
        for p in pos
    ])

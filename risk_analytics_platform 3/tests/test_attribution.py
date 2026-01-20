import numpy as np
import pandas as pd

from src.attribution import (
    component_var_parametric,
    component_var_historical,
    component_var_monte_carlo,
    es_contribution_historical,
)
from src.risk import portfolio_returns


def _make_dummy_returns(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rets = pd.DataFrame(
        {
            "AAA": rng.normal(0, 0.01, size=len(dates)),
            "BBB": rng.normal(0, 0.015, size=len(dates)),
            "CCC": rng.normal(0, 0.02, size=len(dates)),
        },
        index=dates,
    )
    rets.index.name = "date"
    return rets


def test_component_var_sums_to_total_var():
    returns_wide = _make_dummy_returns()
    weights = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    alpha = 0.05

    comp = component_var_parametric(returns_wide, weights, alpha=alpha)
    assert not comp.empty
    total_from_components = float(comp["component_var"].sum())
    assert total_from_components > 0


def test_historical_component_var_positive_total():
    returns_wide = _make_dummy_returns(seed=1)
    weights = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    alpha = 0.05

    comp = component_var_historical(returns_wide, weights, alpha=alpha)
    assert not comp.empty
    assert float(comp["component_var"].sum()) > 0


def test_mc_component_var_positive_total():
    returns_wide = _make_dummy_returns(seed=2)
    weights = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    alpha = 0.05

    comp = component_var_monte_carlo(returns_wide, weights, alpha=alpha, n_sims=5000)
    assert not comp.empty
    assert float(comp["component_var"].sum()) > 0


def test_es_contributions_sum_to_es():
    returns_wide = _make_dummy_returns()
    weights = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3})
    alpha = 0.05

    port = portfolio_returns(returns_wide, weights)
    cutoff = port.quantile(alpha)
    es = -port[port <= cutoff].mean()

    contrib = es_contribution_historical(returns_wide, weights, alpha=alpha)
    assert not contrib.empty
    es_from_contrib = float(contrib["es_contribution"].sum())

    # Allow small numerical differences
    assert abs(es - es_from_contrib) < 1e-8

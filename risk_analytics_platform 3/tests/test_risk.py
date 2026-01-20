import numpy as np
import pandas as pd

from src.risk import var_historical, es_historical, var_parametric_normal


def test_hist_var_es_signs():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, size=2000))
    v = var_historical(rets, alpha=0.05)
    e = es_historical(rets, alpha=0.05)
    assert v > 0
    assert e > 0
    assert e >= v


def test_parametric_var_close_to_theory():
    rng = np.random.default_rng(1)
    sigma = 0.02
    rets = pd.Series(rng.normal(0, sigma, size=10000))
    v = var_parametric_normal(rets, alpha=0.05)
    # For normal: VaR ~ 1.645*sigma
    assert abs(v - 1.645 * sigma) < 0.002

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from risk_engine.attribution import (
    estimate_covariance,
    rolling_volatility_attribution,
    volatility_attribution,
)
from risk_engine.backtest import rolling_backtest
from risk_engine.prices import portfolio_returns


def test_gaussian_uses_sample_covariance_and_euler_volatility():
    returns = np.array([[1, 2], [3, 2], [2, 5]]) * 0.01
    covariance = estimate_covariance(returns, "normal")
    np.testing.assert_allclose(covariance, np.diag([0.0001, 0.0003]), atol=1e-18)

    weights = np.array([0.6, 0.4])
    attribution = volatility_attribution(covariance, weights)
    sigma = sqrt(0.6**2 * 0.0001 + 0.4**2 * 0.0003)
    assert attribution.volatility == pytest.approx(sigma)
    assert attribution.marginal == pytest.approx([0.00006 / sigma, 0.00012 / sigma])
    assert attribution.component == pytest.approx(weights * attribution.marginal)
    assert attribution.component.sum() == pytest.approx(sigma)


def test_ewma_covariance_uses_zero_mean_and_first_outer_product_seed():
    returns = np.array([[1, 2], [2, -1], [0, 1]]) * 0.01
    covariance = estimate_covariance(returns, "ewma", decay=0.5)
    np.testing.assert_allclose(covariance, np.diag([0.000125, 0.000175]), atol=1e-18)
    weights = np.array([0.6, 0.4])
    attribution = volatility_attribution(covariance, weights)
    assert attribution.volatility == pytest.approx(sqrt(weights @ covariance @ weights))
    assert attribution.component.sum() == pytest.approx(attribution.volatility)


def test_long_position_can_reduce_risk_and_marginal_matches_derivative():
    covariance = np.array([[0.04, -0.015], [-0.015, 0.01]])
    weights = np.array([0.8, 0.2])
    attribution = volatility_attribution(covariance, weights)
    assert attribution.component[1] < 0
    assert attribution.component.sum() == pytest.approx(attribution.volatility)

    step = np.array([0, 1e-6])
    upper = volatility_attribution(covariance, weights + step).volatility
    lower = volatility_attribution(covariance, weights - step).volatility
    assert attribution.marginal[1] == pytest.approx((upper - lower) / (2 * step[1]))


def test_zero_and_near_cancelled_variance_leave_contributions_undefined():
    weights = [0.5, 0.5]
    zero = volatility_attribution(np.zeros((2, 2)), weights)
    hedged = volatility_attribution([[1, -1], [-1, 1]], weights)
    nearly_hedged = volatility_attribution([[1, -1 + 1e-15], [-1 + 1e-15, 1]], weights)
    assert zero.volatility == hedged.volatility == 0
    assert nearly_hedged.volatility > 0
    for attribution in (zero, hedged, nearly_hedged):
        assert attribution.marginal is None and attribution.component is None


def test_tiny_well_conditioned_covariance_has_defined_contributions():
    attribution = volatility_attribution(np.diag([1e-300, 4e-300]), [0.5, 0.5])
    assert attribution.volatility == pytest.approx(sqrt(1.25) * 1e-150, rel=1e-12, abs=0)
    assert attribution.component.sum() == pytest.approx(attribution.volatility, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        ([[1]], "one row per weight"),
        ([[1, 1], [0, 1]], "symmetric"),
        ([[1, 2], [2, 1]], "positive semidefinite"),
        ([[1, np.inf], [np.inf, 1]], "finite"),
    ],
)
def test_invalid_covariance_is_rejected(covariance, message):
    with pytest.raises(ValueError, match=message):
        volatility_attribution(covariance, [0.5, 0.5])


def test_rolling_attribution_keeps_zero_volatility_and_undefined_components():
    dates = pd.bdate_range("2024-03-01", periods=4)
    asset = pd.Series([0.01, -0.03, 0.02, -0.06], index=dates)
    assets = pd.DataFrame({"A": asset, "B": -asset})
    weights = {"A": 0.5, "B": 0.5}
    history = rolling_backtest(portfolio_returns(assets, weights), "normal", 2)
    attributed = rolling_volatility_attribution(assets, weights, history, "normal", 2)
    assert len(attributed) == 4
    assert attributed["portfolio_volatility"].eq(0).all()
    assert attributed[["marginal_volatility", "component_volatility"]].isna().all().all()


@pytest.mark.parametrize("method", ["normal", "ewma"])
def test_rolling_attribution_reconciles_and_excludes_target_and_future_returns(method):
    dates = pd.bdate_range("2024-03-01", periods=7)
    assets = pd.DataFrame(
        {"Z": [0.01, -0.02, 0.04, 0.02, -0.06, 0.01, 0.03],
         "A": [0.02, 0.03, -0.01, 0.00, 0.01, -0.02, 0.01]},
        index=dates,
    )
    weights = {"A": 0.25, "Z": 0.75}
    window = 3
    history = rolling_backtest(portfolio_returns(assets, weights), method, window)
    attributed = rolling_volatility_attribution(assets, weights, history, method, window)
    totals = attributed["component_volatility"].groupby(level=0).sum()
    np.testing.assert_allclose(totals, history["volatility"], rtol=1e-12, atol=0)
    assert attributed.loc[dates[3], "forecast_as_of"].eq(dates[2]).all()
    assert attributed.loc[dates[3], "asset"].tolist() == ["Z", "A"]
    assert attributed.loc[dates[3], "weight"].tolist() == [0.75, 0.25]

    changed = assets.copy()
    changed.iloc[4:] += 0.25
    changed_history = rolling_backtest(portfolio_returns(changed, weights), method, window)
    changed_attribution = rolling_volatility_attribution(
        changed, weights, changed_history, method, window
    )
    pd.testing.assert_frame_equal(attributed.loc[:dates[4]], changed_attribution.loc[:dates[4]])
    assert not attributed.loc[dates[5]].equals(changed_attribution.loc[dates[5]])

    wrong_date = history.assign(forecast_as_of=history.index)
    with pytest.raises(ValueError, match="aligned forecasts"):
        rolling_volatility_attribution(assets, weights, wrong_date, method, window)

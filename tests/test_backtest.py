import pandas as pd
import pytest

from risk_engine.backtest import rolling_backtest
from risk_engine.models import METHODS


@pytest.mark.parametrize("method", METHODS)
def test_realized_and_future_returns_cannot_change_forecasts(method):
    dates = pd.date_range("2024-01-02", periods=7)
    returns = pd.Series([0.01, -0.03, 0.02, 0.001, -0.02, -0.03, 0.04], index=dates)
    shifted = returns.copy()
    shifted.iloc[3:] = -0.60
    baseline = rolling_backtest(returns, method, window=3)
    changed = rolling_backtest(shifted, method, window=3)

    assert baseline.index.tolist() == dates[3:].tolist()
    assert baseline["forecast_as_of"].tolist() == dates[2:-1].tolist()
    forecast_columns = ["var", "es", "volatility"]
    pd.testing.assert_series_equal(
        baseline.loc[dates[3], forecast_columns], changed.loc[dates[3], forecast_columns]
    )
    assert changed.loc[dates[3], "loss"] == 0.60
    assert bool(changed.loc[dates[3], "breach"])
    assert changed.loc[dates[4], "var"] > baseline.loc[dates[4], "var"]


def test_var_equality_is_not_a_breach_and_window_rolls():
    dates = pd.date_range("2024-01-02", periods=6)
    returns = pd.Series([-0.20, -0.03, 0.01, -0.03, -0.04, -0.02], index=dates)
    history = rolling_backtest(returns, "historical", window=2)
    assert history["var"].tolist() == pytest.approx([0.20, 0.03, 0.03, 0.04])
    assert history["breach"].tolist() == [False, False, True, False]


def test_rejects_reversed_dates_and_insufficient_returns():
    returns = pd.Series([0.1, 0.2, -0.1], index=pd.date_range("2024-01-02", periods=3))
    with pytest.raises(ValueError, match="increasing"):
        rolling_backtest(returns.iloc[::-1], "ewma", window=2)
    with pytest.raises(ValueError, match="at least 4 returns"):
        rolling_backtest(returns, "ewma", window=3)

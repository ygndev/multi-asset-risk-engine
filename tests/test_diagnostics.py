import pandas as pd
import pytest

from risk_engine.diagnostics import max_drawdown, var_exceedances


def test_tail_diagnostics_pair_same_day_losses_and_es_and_keep_trading_runs():
    dates = pd.bdate_range("2024-03-01", periods=6, name="target_date")
    history = pd.DataFrame(
        {
            "forecast_as_of": dates - pd.offsets.BDay(1),
            "loss": [0.02, 0.03, 0.01, 0.05, 0.01, 0.00],
            "var": [0.01] * 6,
            "es": [0.015, 0.03, 0.7, 0.055, 0.8, 0.99],
        },
        index=dates,
    )
    tail = var_exceedances(history)
    assert tail.index.tolist() == [dates[0], dates[1], dates[3]]
    assert tail["run_id"].tolist() == [1, 1, 2]
    assert tail.loc[dates[1]:, "run_id"].tolist() == [1, 2]
    assert tail["excess_over_var"].tolist() == pytest.approx([0.01, 0.02, 0.04])
    assert tail["loss_minus_es"].tolist() == pytest.approx([0.005, 0.0, -0.005])
    assert tail["es"].mean() == pytest.approx(0.1 / 3)
    assert var_exceedances(history.assign(loss=0.01)).empty


def test_drawdown_compounds_returns_and_identifies_peak_and_trough():
    dates = pd.bdate_range("2024-01-02", periods=5)
    period = pd.Series([0.10, -0.10, -0.20, 0.06], index=dates[1:])
    drawdown = max_drawdown(period, dates[0])
    assert drawdown.magnitude == pytest.approx(0.28)
    assert (drawdown.peak_date, drawdown.trough_date) == (dates[1], dates[3])


def test_drawdown_includes_preperiod_baseline_and_handles_no_decline():
    dates = pd.bdate_range("2024-01-02", periods=3)
    drawdown = max_drawdown(pd.Series([-0.10, 0.01], index=dates[1:]), dates[0])
    assert drawdown.magnitude == pytest.approx(0.10)
    assert (drawdown.peak_date, drawdown.trough_date) == (dates[0], dates[1])
    gains = max_drawdown(pd.Series([0.01, 0.02], index=dates[1:]), dates[0])
    assert gains.magnitude == 0
    assert gains.peak_date is None and gains.trough_date is None

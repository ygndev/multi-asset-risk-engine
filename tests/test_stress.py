import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from risk_engine.attribution import volatility_attribution
from risk_engine.cli import main
from risk_engine.prices import portfolio_returns, portfolio_weights, price_returns
from risk_engine.stress import PORTFOLIOS, parse_scenarios, pre_stress_covariances, replay_period


def test_historical_replay_compounds_daily_rebalanced_returns_and_recovery():
    dates = pd.bdate_range("2024-01-02", periods=5)
    assets = pd.DataFrame(
        {"A": [0.9, 0.2, 0.0, 0.25, 0.0], "B": [0.8, 0.0, -0.2, 0.25, 0.0]},
        index=dates,
    )
    returns = portfolio_returns(assets, {"A": 0.5, "B": 0.5})
    period = slice(dates[1], dates[2])
    replay = replay_period(returns, period)
    assert replay.path["portfolio_return"].tolist() == pytest.approx([0.10, -0.10])
    assert replay.path["cumulative_return"].tolist() == pytest.approx([0.10, -0.01])
    assert replay.drawdown.magnitude == pytest.approx(0.10)
    assert replay.drawdown.peak_date == dates[1]
    assert replay.drawdown.trough_date == replay.path["portfolio_return"].idxmin() == dates[2]
    assert replay.recovery_date == dates[3]
    assert replay.recovery_sessions == 1

    changed = returns.copy()
    changed.iloc[3:] = -0.10
    censored = replay_period(changed, period)
    pd.testing.assert_frame_equal(censored.path, replay.path)
    assert censored.drawdown == replay.drawdown
    assert censored.recovery_date is None
    assert censored.recovery_sessions == 2


def test_drawdown_includes_the_pre_window_close_and_can_recover_in_window():
    dates = pd.bdate_range("2024-01-02", periods=4)
    returns = pd.Series([0.5, -0.2, 0.25, 0.03], index=dates)
    replay = replay_period(returns, slice(dates[1], dates[3]))
    assert replay.drawdown.magnitude == pytest.approx(0.20)
    assert replay.drawdown.peak_date == dates[0]
    assert replay.recovery_date == dates[2]
    assert replay.recovery_sessions == 1
    assert replay.path["drawdown"].iloc[-1] == 0

    with pytest.raises(ValueError, match="preceding session"):
        replay_period(returns, slice(dates[0], dates[2]))
    with pytest.raises(ValueError, match="chronological"):
        replay_period(returns.iloc[::-1], slice(dates[1], dates[3]))


def test_positive_path_has_no_drawdown_recovery():
    dates = pd.bdate_range("2024-01-02", periods=3)
    replay = replay_period(pd.Series([0, 0.01, 0.02], index=dates), slice(dates[1], None))
    assert replay.drawdown.magnitude == 0
    assert replay.path["cumulative_return"].iloc[-1] == pytest.approx(0.0302)
    assert replay.recovery_date is None
    assert replay.recovery_sessions is None


def test_recovery_accepts_a_repeated_close_despite_compounding_roundoff():
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"IVV": [100.0, 100.0, 82.0, 100.0]}, index=dates)
    returns = price_returns(prices)["IVV"]
    replay = replay_period(returns, slice(dates[2], dates[3]))
    assert replay.recovery_date == dates[3]
    assert replay.recovery_sessions == 1


def test_replay_rejects_noncontiguous_windows():
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.Series([0, 0.01, -0.20, 0.03, 0.04], index=dates)
    with pytest.raises(ValueError, match="contiguous"):
        replay_period(returns, slice(dates[1], dates[-1], 2))


def test_percentage_shocks_are_converted_by_label_and_weighted():
    config = b'''{"unit":"percent","scenarios":[
        {"name":"Selloff","shocks":{"IAU":4,"IEF":2,"IVV":-15}},
        {"name":"Small","shocks":{"IVV":-0.05,"IEF":0,"IAU":-100}}
    ]}'''
    shocks = parse_scenarios(config)
    assert shocks.columns.tolist() == ["IVV", "IEF", "IAU"]
    assert shocks.loc["Small"].tolist() == pytest.approx([-0.0005, 0, -1])
    pnl = {
        name: portfolio_returns(shocks[["IAU", "IVV", "IEF"]], weights).at["Selloff"]
        for name, weights in PORTFOLIOS.items()
    }
    assert pnl == pytest.approx({"Baseline": -0.061, "Treasury tilt": -0.027, "Equity tilt": -0.095})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (b'{"scenarios":[]}', "fields unit and scenarios"),
        (b'{"unit":"decimal","scenarios":[]}', 'unit must be "percent"'),
        (b'{"unit":"percent","scenarios":[]}', "nonempty list"),
        (
            b'{"unit":"percent","scenarios":[{"name":"A","shocks":{"IVV":1,"IVV":2,"IEF":0,"IAU":0}}]}',
            "duplicate scenario JSON field: IVV",
        ),
        (
            b'{"unit":"percent","scenarios":[{"name":"A","shocks":{"IVV":1,"IEF":0,"GLD":0}}]}',
            "must specify IVV, IEF and IAU",
        ),
    ],
)
def test_invalid_scenario_structure_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        parse_scenarios(config)


@pytest.mark.parametrize(
    ("shock", "message"),
    [
        (True, "numeric percent"),
        ("-2%", "numeric percent"),
        (None, "numeric percent"),
        (-100.01, "at least -100%"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_invalid_asset_shocks_are_rejected(shock, message):
    config = {"unit": "percent", "scenarios": [
        {"name": "Bad IVV", "shocks": {"IVV": shock, "IEF": 1, "IAU": 2}}
    ]}
    with pytest.raises(ValueError, match=message):
        parse_scenarios(json.dumps(config).encode())


def test_duplicate_scenario_names_are_rejected():
    shocks = {"IVV": 0, "IEF": 1, "IAU": 2}
    config = {"unit": "percent", "scenarios": [
        {"name": "A", "shocks": shocks}, {"name": " A ", "shocks": shocks}
    ]}
    with pytest.raises(ValueError, match="duplicate scenario name"):
        parse_scenarios(json.dumps(config).encode())


def test_sensitivity_uses_only_pre_window_returns_and_aligns_weights():
    dates = pd.bdate_range("2019-01-02", periods=254)
    x = np.arange(len(dates))
    assets = pd.DataFrame(
        {"IAU": 0.02 * np.cos(x), "IVV": 0.01 * np.sin(x), "IEF": 0.006 * np.cos(3 * x)},
        index=dates,
    )
    target = dates[252]
    with pytest.raises(ValueError, match="chronological"):
        pre_stress_covariances(assets.iloc[::-1], target)
    with pytest.raises(ValueError, match="unique"):
        pre_stress_covariances(pd.concat([assets.iloc[:1], assets]), target)
    as_of, covariances = pre_stress_covariances(assets, target)
    assert as_of == dates[251]
    baseline = PORTFOLIOS["Baseline"]
    weights = portfolio_weights(assets.columns, baseline)
    assert weights.tolist() == [0.2, 0.5, 0.3]
    assert volatility_attribution(covariances["normal"], weights).volatility == pytest.approx(
        portfolio_returns(assets.iloc[2:252], baseline).std(ddof=1)
    )
    changed = assets.copy()
    changed.iloc[:2] = 2
    changed.iloc[252:] = 3
    _, changed_covariances = pre_stress_covariances(changed, target)
    for method in covariances:
        np.testing.assert_array_equal(covariances[method], changed_covariances[method])

    reordered = assets[["IEF", "IAU", "IVV"]]
    _, reordered_covariances = pre_stress_covariances(reordered, target)
    for allocation in PORTFOLIOS.values():
        weights = portfolio_weights(assets.columns, allocation)
        reordered_weights = portfolio_weights(reordered.columns, allocation)
        for method in covariances:
            sigma = volatility_attribution(covariances[method], weights).volatility
            reordered_sigma = volatility_attribution(
                reordered_covariances[method], reordered_weights
            ).volatility
            assert sigma == pytest.approx(reordered_sigma)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"IVV": 0.7, "IEF": 0.2, "IAU": 0.2}, "sum to one"),
        ({"IVV": 0.8, "IEF": 0.3, "IAU": -0.1}, "nonnegative"),
        ({"IVV": 0.5, "IEF": 0.3, "GLD": 0.2}, "symbols must match"),
        ({"IVV": 0.5, "IEF": 0.3, "IAU": np.nan}, "finite"),
    ],
)
def test_invalid_predefined_weights_fail_portfolio_validation(weights, message):
    with pytest.raises(ValueError, match=message):
        portfolio_weights(pd.Index(["IVV", "IEF", "IAU"]), weights)


def test_stress_cli_publishes_replay_and_hypothetical_outputs(monkeypatch, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    dates = pd.bdate_range("2019-01-01", "2023-01-06", name="date")
    prices = pd.DataFrame(100.0, index=dates, columns=["IVV", "IEF", "IAU"])
    prices.loc["2020-03-02":, "IVV"] = 80
    prices.loc["2022-01-03":, "IVV"] = 72
    provenance = {"source": "offline fixture", "prices_sha256": "fixture-checksum"}
    monkeypatch.setattr("risk_engine.stress.read_market_snapshot", lambda _: (prices, provenance))
    config = b'{"unit":"percent","scenarios":[{"name":"IVV drops","shocks":{"IVV":-20,"IEF":0,"IAU":0}}]}'
    scenario_file = tmp_path / "custom.json"
    scenario_file.write_bytes(config)
    output = tmp_path / "stress"
    main(["stress-test", str(tmp_path / "snapshot"), str(scenario_file), str(output)])
    assert "Saved stress study" in capsys.readouterr().out
    assert (output / "scenarios.json").read_bytes() == config
    report = (output / "stress-report.md").read_text(encoding="utf-8")
    assert "fixture-checksum" in report and sha256(config).hexdigest() in report
    assert "| March 2020 | Baseline | 22 | 10.00% | 10.00%" in report
    assert "| March 2020 | Baseline | 2020-02-28 | 0.000% | 0.000% | 10.00% |" in report
    assert "| IVV drops | -20% | +0% | +0% | -10% | -6% | -14% |" in report
    paths = pd.read_csv(output / "historical-paths.csv")
    march = paths[(paths["period"] == "march-2020") & (paths["portfolio"] == "Baseline")]
    assert march["date"].iloc[0] == "2020-03-02"
    assert march["portfolio_return"].iloc[0] == pytest.approx(-0.10)
    assert march["cumulative_return"].iloc[-1] == pytest.approx(-0.10)
    assert (output / "stress-paths.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_stress_cli_reports_invalid_units_without_publishing_output(tmp_path, capsys):
    scenario_file = tmp_path / "invalid.json"
    scenario_file.write_bytes(b'{"unit":"decimal","scenarios":[]}')
    output = tmp_path / "stress"
    with pytest.raises(SystemExit) as failure:
        main(["stress-test", str(tmp_path / "snapshot"), str(scenario_file), str(output)])
    assert failure.value.code == 2
    assert 'unit must be "percent"' in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [scenario_file]

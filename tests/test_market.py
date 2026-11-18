import json
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from risk_engine.cli import main
from risk_engine.market import (
    SNAPSHOT_SPEC,
    expected_sessions,
    fetch_market_prices,
    read_market_snapshot,
    validate_market_prices,
)
from risk_engine.models import METHODS


@pytest.fixture
def prices():
    dates = pd.DatetimeIndex(["2024-12-26", "2024-12-27", "2024-12-30", "2024-12-31"])
    return pd.DataFrame(
        {
            "IVV": [100.0, 101.0, 99.0, 102.0],
            "IEF": [95.0, 96.0, 96.0, 95.0],
            "IAU": [30.0, 31.0, 30.0, 32.0],
        },
        index=dates,
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda p: p.iloc[::-1], "chronological order"),
        (lambda p: p.iloc[[0, 1, 1, 2, 3]], "duplicate market date"),
        (lambda p: p.iloc[1:-1], "2 missing sessions"),
        (lambda p: p.drop(p.index[1]), "1 missing sessions.*2024-12-27"),
        (
            lambda p: p.rename(index={p.index[2]: pd.Timestamp("2024-12-28")}),
            "1 unexpected dates.*2024-12-28",
        ),
        (lambda p: p.drop(columns="IEF"), "columns must be exactly"),
        (lambda p: p.assign(IEF=[95.0, np.nan, 96.0, 95.0]), "IEF on 2024-12-27"),
        (lambda p: p.assign(IAU=[0.0, 31.0, 30.0, 32.0]), "IAU on 2024-12-26"),
        (lambda p: p.assign(IVV=[100.0, 101.0, np.inf, 102.0]), "IVV on 2024-12-30"),
    ],
)
def test_market_validation_rejects_invalid_sessions_and_prices(prices, change, error):
    validate_market_prices(prices, prices.index)
    with pytest.raises(ValueError, match=error):
        validate_market_prices(change(prices), prices.index)


def test_full_calendar_includes_special_market_closures():
    pytest.importorskip("exchange_calendars")
    sessions = expected_sessions()
    assert sessions[0] == pd.Timestamp("2010-01-04")
    assert sessions[-1] == pd.Timestamp("2024-12-31")
    for closed in ("2012-10-29", "2012-10-30", "2018-12-05", "2022-06-20"):
        assert pd.Timestamp(closed) not in sessions
    assert pd.Timestamp("2020-03-16") in sessions


def test_fetch_writes_checked_snapshot(prices, monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")
    xcals = pytest.importorskip("exchange_calendars")

    def download(symbols, **options):
        assert symbols == ["IVV", "IEF", "IAU"]
        assert options["start"] == "2010-01-01"
        assert options["end"] == "2025-01-01"
        assert options["interval"] == "1d"
        assert options["auto_adjust"] is True
        assert options["keepna"] is True
        return pd.concat({"Close": prices[["IEF", "IAU", "IVV"]]}, axis=1)

    monkeypatch.setattr(yf, "download", download)
    monkeypatch.setattr("risk_engine.market.expected_sessions", lambda: prices.index)
    snapshot = tmp_path / "snapshot"
    main(["fetch", str(snapshot)])
    csv_path = snapshot / "prices.csv"
    saved, provenance = read_market_snapshot(snapshot)
    pd.testing.assert_frame_equal(saved, prices.rename_axis("date"), check_index_type=False)
    assert provenance["yfinance_version"] == yf.__version__
    assert provenance["exchange_calendars_version"] == xcals.__version__
    assert provenance["retrieved_at_utc"].endswith("+00:00")
    assert provenance["observations"] == 4
    assert provenance["prices_sha256"] == sha256(csv_path.read_bytes()).hexdigest()

    with pytest.raises(FileExistsError, match="output already exists"):
        fetch_market_prices(snapshot)

    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        read_market_snapshot(snapshot)


@pytest.fixture
def snapshot_file(prices, monkeypatch, tmp_path):
    monkeypatch.setattr("risk_engine.market.expected_sessions", lambda: prices.index)
    csv_path = tmp_path / "prices.csv"
    prices.to_csv(csv_path, index_label="date", lineterminator="\n")
    provenance = {
        **SNAPSHOT_SPEC,
        "retrieved_at_utc": "2025-01-01T10:00:00+00:00",
        "yfinance_version": "1.8.0",
        "exchange_calendars_version": "4.13.3",
        "first_session": prices.index[0].date().isoformat(),
        "last_session": prices.index[-1].date().isoformat(),
        "observations": len(prices),
        "prices_sha256": sha256(csv_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return csv_path


def test_snapshot_parses_the_same_bytes_that_were_hashed(snapshot_file, prices, monkeypatch):
    replacement = prices.assign(IVV=777.0)

    def replace_after_read(data):
        digest = sha256(data)
        replacement.to_csv(snapshot_file, index_label="date", lineterminator="\n")
        return digest

    monkeypatch.setattr("risk_engine.market.sha256", replace_after_read)
    verified, _ = read_market_snapshot(snapshot_file.parent)
    assert verified["IVV"].tolist() == prices["IVV"].tolist()
    assert pd.read_csv(snapshot_file)["IVV"].eq(777.0).all()


@pytest.mark.parametrize("timestamp", ["yesterday", "2025-01-01T10:00:00", "2025-01-01T10:00:00+02:00"])
def test_snapshot_rejects_invalid_utc_provenance(snapshot_file, timestamp):
    metadata_path = snapshot_file.with_name("provenance.json")
    provenance = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance["retrieved_at_utc"] = timestamp
    metadata_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="ISO 8601 UTC timestamp"):
        read_market_snapshot(snapshot_file.parent)


@pytest.mark.parametrize("response", ["empty", "partial", "network error"])
def test_incomplete_download_does_not_publish_snapshot(response, prices, monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")

    def download(*args, **kwargs):
        if response == "empty":
            return pd.DataFrame()
        if response == "network error":
            raise OSError("offline")
        return pd.concat({"Close": prices.assign(IEF=np.nan)}, axis=1)

    monkeypatch.setattr(yf, "download", download)
    monkeypatch.setattr("risk_engine.market.expected_sessions", lambda: prices.index)
    with pytest.raises(ValueError, match="Yahoo Finance|IEF on"):
        fetch_market_prices(tmp_path / "snapshot")
    assert list(tmp_path.iterdir()) == []


def test_market_study_reports_full_history_and_stress_subsets(monkeypatch, tmp_path):
    yf = pytest.importorskip("yfinance")
    pytest.importorskip("exchange_calendars")
    pytest.importorskip("matplotlib")
    dates = expected_sessions()
    price = pd.Series(100.0, index=dates)
    price.loc["2020-03-02":] = 90.0
    price.loc["2022-01-03":] = 85.5
    adjusted = pd.DataFrame({"IVV": price, "IEF": price, "IAU": price})
    response = pd.concat({"Close": adjusted}, axis=1)
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: response)
    snapshot = tmp_path / "data"
    output = tmp_path / "report"
    main(["fetch", str(snapshot)])
    main(["market-study", str(snapshot), str(output)])

    forecasts = pd.read_csv(output / "forecasts.csv", parse_dates=["target_date", "forecast_as_of"])
    assert len(forecasts) == 3 * (len(dates) - 251)
    for method in METHODS:
        history = forecasts[forecasts["method"] == method].set_index("target_date")
        assert history.index.equals(dates[251:])
        assert history["forecast_as_of"].tolist() == dates[250:-1].tolist()
        assert history.loc["2020-03-02", "loss"] == pytest.approx(0.10)
        assert history.loc["2022-01-03", "loss"] == pytest.approx(0.05)
        assert history.loc["2020-03-02", "var"] == 0
        assert history.loc["2020-03-02", "es"] == 0
        assert history.loc["2020-03-02", "forecast_as_of"] == pd.Timestamp("2020-02-28")
        assert history.index[history["breach"]].tolist() == list(
            pd.to_datetime(["2020-03-02", "2022-01-03"])
        )

    report = (output / "report.md").read_text(encoding="utf-8")
    provenance = json.loads((snapshot / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["prices_sha256"] in report
    assert "## March 2020" in report
    assert "22 forecasts per model" in report
    assert "## Calendar year 2022" in report
    assert "251 forecasts per model" in report
    assert "Largest realized portfolio loss: 10.00% on 2020-03-02" in report
    for name in ("full-period", "march-2020", "2022"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    validation = (output / "model-validation.md").read_text(encoding="utf-8")
    assert provenance["prices_sha256"] in validation
    assert "Maximum drawdown 10.00% (2020-02-28 to 2020-03-02)" in validation
    assert "The average paired gap is descriptive" in validation
    march = validation.split("### March 2020")[1].split("### Calendar year 2022")[0]
    assert "| Historical | 1 | 10.000% | 0.000% | +10.000% |" in march

    tail = pd.read_csv(output / "var-exceedances.csv")
    assert len(tail) == 6
    assert tail["target_date"].tolist() == ["2020-03-02", "2022-01-03"] * 3
    assert tail["run_id"].tolist() == [1, 2] * 3
    assert sorted(tail["loss_minus_es"]) == pytest.approx([0.05] * 3 + [0.10] * 3)

    attribution = pd.read_csv(output / "volatility-attribution.csv")
    assert len(attribution) == 2 * 3 * (len(dates) - 251)
    before_shock = attribution[attribution["target_date"] == "2020-03-02"]
    assert before_shock["component_volatility"].isna().all()
    assert before_shock["portfolio_volatility"].eq(0).all()
    defined = attribution.dropna(subset=["component_volatility"])
    groups = defined.groupby(["method", "target_date"])
    np.testing.assert_allclose(
        groups["component_volatility"].sum(), groups["portfolio_volatility"].first(),
        rtol=1e-8, atol=1e-14,
    )

import pytest

from risk_engine.cli import main


def test_demo_reports_each_model_and_out_of_sample_count(capsys):
    main(["demo"])
    report = capsys.readouterr().out
    assert "PCG64 seed 42" in report
    assert "(1,250 per model)" in report
    assert "| Historical |" in report
    assert "| Normal |" in report
    assert "| EWMA normal |" in report
    assert "Christoffersen conditional coverage LR / p" in report


@pytest.mark.parametrize("equity_symbol", ["EQ", "EQ=USD"])
def test_run_reads_an_adjusted_close_csv(tmp_path, capsys, equity_symbol):
    prices = tmp_path / "portfolio.csv"
    prices.write_text(
        f"date,{equity_symbol},BOND\n"
        "2024-01-02,100,100\n2024-01-03,101,102\n"
        "2024-01-04,100,102\n2024-01-05,101,101\n",
        encoding="utf-8",
    )
    weights = ["BOND=0.30", f"{equity_symbol}=0.70"]
    main(["run", str(prices), "--weights", *weights, "--window", "2"])
    report = capsys.readouterr().out
    assert "Adjusted-close CSV (portfolio.csv)" in report
    assert f"BOND 30.00%, {equity_symbol} 70.00%" in report
    assert "(1 per model)" in report
    assert "N/A | N/A" in report


def test_nominal_breach_rate_formats_small_nonzero_values(capsys):
    main(["demo", "--confidence", "0.9999999"])
    assert "Nominal breach rate: 1e-05%." in capsys.readouterr().out


def test_invalid_confidence_fails_without_a_report(capsys):
    with pytest.raises(SystemExit) as failure:
        main(["demo", "--confidence", "1"])
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert "confidence" in captured.err
    assert captured.out == ""

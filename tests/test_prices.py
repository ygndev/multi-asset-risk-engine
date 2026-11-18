import pandas as pd
import pytest

from risk_engine.prices import portfolio_returns, price_returns, read_prices


def test_prices_sorted_and_weights_matched_by_symbol(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,EQ,BOND_ETF\n2024-01-04,99,102\n2024-01-02,100,100\n2024-01-03,110,100\n",
        encoding="utf-8",
    )
    prices = read_prices(path)
    asset_returns = price_returns(prices)
    portfolio = portfolio_returns(asset_returns, {"BOND_ETF": 0.25, "EQ": 0.75})
    assert portfolio.index.name == "date"
    assert portfolio.index.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04"]
    assert portfolio.tolist() == pytest.approx([0.075, -0.07])


@pytest.mark.parametrize(
    ("csv_text", "error"),
    [
        ("date,EQ,EQ\n2024-01-02,10,11\n", "symbols"),
        ("date,EQ\n2024-01-02,10\n2024-01-02,11\n", "duplicate price date"),
        ("date,EQ\n20240102,10\n", "YYYY-MM-DD"),
        ("date,EQ\n2024-02-30,10\n", "YYYY-MM-DD"),
        ("date,EQ\n2024-01-02,\n", "finite, positive price"),
        ("date,EQ\n2024-01-02,NaN\n", "finite, positive price"),
        ("date,EQ\n2024-01-02,0\n", "finite, positive price"),
        ("date,EQ\n2024-01-02,10,11\n", "expected 2 fields"),
        ("date,EQ\n2024-01-02,10\n", "at least two"),
    ],
)
def test_invalid_price_csv(tmp_path, csv_text, error):
    path = tmp_path / "prices.csv"
    path.write_text(csv_text, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        read_prices(path)


@pytest.mark.parametrize(
    ("weights", "error"),
    [
        ({"EQ": 1}, "symbols"),
        ({"EQ": 1.1, "BOND": -0.1}, "nonnegative"),
        ({"EQ": 0.50, "BOND": 0.40}, "sum"),
    ],
)
def test_invalid_portfolio_weights(weights, error):
    returns = pd.DataFrame({"EQ": [0.01], "BOND": [0.02]})
    with pytest.raises(ValueError, match=error):
        portfolio_returns(returns, weights)

import argparse
from pathlib import Path

from risk_engine.backtest import rolling_backtest
from risk_engine.demo import WEIGHTS, synthetic_prices
from risk_engine.market_study import _write_plot
from risk_engine.models import METHODS
from risk_engine.prices import portfolio_returns, price_returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the synthetic VaR and ES figure")
    parser.add_argument("output", type=Path, help="PNG output path")
    args = parser.parse_args()

    returns = portfolio_returns(price_returns(synthetic_prices()), WEIGHTS)
    histories = {method: rolling_backtest(returns, method, 250, 0.99, 0.94) for method in METHODS}
    _write_plot("Seeded synthetic demo", histories, args.output)


if __name__ == "__main__":
    main()

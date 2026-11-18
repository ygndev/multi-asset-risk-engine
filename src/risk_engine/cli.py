import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .backtest import rolling_backtest
from .demo import SEED, WEIGHTS, synthetic_prices
from .models import METHODS, estimate_risk
from .prices import portfolio_returns, price_returns, read_prices
from .validation import LikelihoodRatio, coverage_tests

METHOD_LABELS = {
    "historical": "Historical",
    "normal": "Normal",
    "ewma": "EWMA normal",
}


def _parse_weights(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        symbol, _, value = entry.rpartition("=")
        symbol = symbol.strip()
        if not symbol or not value or symbol in weights:
            raise ValueError("supply each weight once using SYMBOL=WEIGHT")
        try:
            weights[symbol] = float(value)
        except ValueError:
            raise ValueError(f"invalid weight for {symbol}: {value}") from None
    return weights


def _format_test(test: LikelihoodRatio | None) -> str:
    if test is None:
        return "N/A"
    return f"{test.statistic:.4f} / {test.p_value:.4g}"


def _render_report(
    returns: pd.Series,
    weights: dict[str, float],
    source: str,
    window: int,
    confidence: float,
    decay: float,
) -> str:
    allocations = ", ".join(f"{asset} {weight:.2%}" for asset, weight in weights.items())
    count = len(returns) - window
    forecasts = []
    tests = []
    for method in METHODS:
        label = METHOD_LABELS[method]
        history = rolling_backtest(returns, method, window, confidence, decay)
        risk = estimate_risk(returns.iloc[-window:], method, confidence, decay)
        forecasts.append(
            f"| {label} | {risk.var:.4%} | {risk.es:.4%} | {risk.volatility:.4%} |"
        )
        hits = history["breach"]
        validation = coverage_tests(hits, confidence)
        transitions = " / ".join(map(str, validation.transitions))
        tests.append(
            f"| {label} | {int(hits.sum())} | {hits.mean():.2%} | {transitions} | "
            f"{_format_test(validation.kupiec)} | "
            f"{_format_test(validation.independence)} | "
            f"{_format_test(validation.conditional)} |"
        )

    return "\n".join(
        [
            "# Portfolio risk report",
            "",
            f"- Source: {source}",
            f"- Portfolio: {allocations}",
            f"- As of: {returns.index[-1].date()}",
            "- Horizon: one observation step",
            f"- Confidence: {confidence}",
            f"- Lookback: {window} returns",
            f"- EWMA decay: {decay}",
            "",
            "## Next-observation forecast",
            "",
            "| Method | Loss VaR | Loss ES | Volatility |",
            "| --- | ---: | ---: | ---: |",
            *forecasts,
            "",
            "## Out-of-sample VaR backtest",
            "",
            f"{returns.index[window].date()} to {returns.index[-1].date()} "
            f"({count:,} per model). "
            f"Nominal breach rate: {(1 - confidence) * 100:.4g}%.",
            "",
            "| Method | Breaches | Breach rate | Transitions (00/01/10/11) | "
            "Kupiec LR / p | Christoffersen independence LR / p | "
            "Christoffersen conditional coverage LR / p |",
            "| --- | ---: | ---: | --- | --- | --- | --- |",
            *tests,
            "",
            "A breach means realized loss strictly exceeds VaR. Sparse transitions make "
            "asymptotic p-values less reliable. N/A indicates an absent transition state. "
            "The likelihood-ratio tests assess VaR coverage and dependence.",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-period portfolio risk and VaR backtesting")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the seeded synthetic example")
    run = commands.add_parser("run", help="analyze a wide adjusted-close CSV")
    run.add_argument("prices", type=Path)
    run.add_argument("--weights", nargs="+", required=True, metavar="SYMBOL=WEIGHT")
    for command in (demo, run):
        command.add_argument("--window", type=int, default=250, help="lookback in returns")
        command.add_argument("--confidence", type=float, default=0.99)
        command.add_argument("--decay", type=float, default=0.94, help="EWMA decay")
    args = parser.parse_args(argv)

    try:
        if args.command == "demo":
            prices, weights = synthetic_prices(), WEIGHTS
            source = f"Synthetic prices (PCG64 seed {SEED}; weekday calendar)"
        else:
            prices, weights = read_prices(args.prices), _parse_weights(args.weights)
            source = f"Adjusted-close CSV ({args.prices.name})"
        returns = portfolio_returns(price_returns(prices), weights)
        report = _render_report(returns, weights, source, args.window, args.confidence, args.decay)
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
    print(report)

from pathlib import Path

import pandas as pd

from .backtest import rolling_backtest
from .market import WEIGHTS, new_output_directory, read_market_snapshot
from .models import METHODS
from .prices import portfolio_returns, price_returns
from .validation import coverage_tests

WINDOW = 250
CONFIDENCE = 0.99
DECAY = 0.94
METHOD_LABELS = {
    "historical": "Historical",
    "normal": "Gaussian",
    "ewma": "EWMA-Gaussian",
}
PERIODS = (
    ("Full backtest", "full-period", slice(None)),
    ("March 2020", "march-2020", slice("2020-03-01", "2020-03-31")),
    ("Calendar year 2022", "2022", slice("2022-01-01", "2022-12-31")),
)


def run_market_study(snapshot: Path, destination: Path) -> None:
    """Evaluate fixed models on a verified price snapshot and publish the report."""
    with new_output_directory(destination) as output:
        prices, provenance = read_market_snapshot(snapshot)
        returns = portfolio_returns(price_returns(prices), WEIGHTS)
        histories = {
            method: rolling_backtest(returns, method, WINDOW, CONFIDENCE, DECAY)
            for method in METHODS
        }
        pd.concat(histories, names=["method"]).to_csv(
            output / "forecasts.csv", date_format="%Y-%m-%d", lineterminator="\n"
        )
        report = [
            "# Historical market risk study",
            "",
            f"- Source: {provenance['source']} (yfinance {provenance['yfinance_version']})",
            f"- Retrieved (UTC): {provenance['retrieved_at_utc']}",
            f"- Adjustments: {provenance['adjustment']}",
            f"- Price sessions: {prices.index[0].date()} through {prices.index[-1].date()} "
            f"({len(prices):,} common sessions, XNYS calendar, "
            f"exchange-calendars {provenance['exchange_calendars_version']})",
            f"- Price snapshot SHA-256: `{provenance['prices_sha256']}`",
            "- Portfolio: IVV 50%, IEF 30%, IAU 20%, in USD",
            f"- Forecast: one trading session, {WINDOW} prior returns, "
            f"{CONFIDENCE:.0%} confidence, EWMA decay {DECAY}",
            "",
            "The portfolio rebalances at each session boundary at fixed weights with "
            "zero transaction costs. Loss is the negative portfolio return. Each forecast "
            "uses returns available through the preceding session. Both stress periods reuse "
            "the corresponding forecasts from the full backtest.",
            "",
            "The [forecast CSV](forecasts.csv) includes each target date, forecast date, "
            "realized loss, VaR, ES, volatility and breach indicator. Values are fractions "
            "of portfolio value. A breach means loss strictly exceeds VaR.",
        ]
        for title, filename, dates in PERIODS:
            period = {method: history.loc[dates] for method, history in histories.items()}
            report.extend(_summarize_period(title, filename, period))
            _write_plot(title, period, output / f"{filename}.png")
        report.extend([
            "",
            "## Interpretation",
            "",
            "- Historical simulation uses the empirical tail. At this window and confidence, "
            "historical ES uses 2.5 observations' worth of tail mass. Even with independent "
            "continuous losses, rank granularity gives the historical VaR a marginal "
            "predictive breach rate of 3/251 (about 1.195%). The tests use the nominal 1%.",
            "- The Gaussian model estimates the mean and standard deviation over the window. "
            "EWMA-Gaussian assumes zero mean and initializes variance with the first squared "
            "return in each window. Both use Gaussian tails. Estimated volatility and the "
            "mean assumption can both contribute to differences in their forecasts.",
            "- Kupiec assesses unconditional VaR coverage. Christoffersen assesses "
            "first-order breach dependence and joint coverage, conditioning on the first "
            "indicator in each period. N/A means an origin or destination state is absent.",
            "- The p-values use asymptotic chi-square approximations. Short stress periods "
            "have few expected breaches at 99% confidence, and sparse transitions limit "
            "power and the reliability of those approximations. The stress periods are "
            "subsets of the same backtest and provide descriptive comparisons.",
            "- The likelihood-ratio tests evaluate VaR. The CSV and plots show ES forecasts "
            "for comparison, and ES calibration requires its own test.",
            "",
        ])
        (output / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")


def _summarize_period(
    title: str, filename: str, histories: dict[str, pd.DataFrame]
) -> list[str]:
    loss = histories["historical"]["loss"]
    worst_date = loss.idxmax()
    lines = [
        "",
        f"## {title}",
        "",
        f"{loss.index[0].date()} through {loss.index[-1].date()} · "
        f"{len(loss):,} forecasts per model · "
        f"{len(loss) * (1 - CONFIDENCE):.2f} nominal expected breaches per model.",
        f"Largest realized portfolio loss: {loss.at[worst_date]:.2%} on {worst_date.date()}.",
        "",
        "| Model | Breaches | Rate | Transitions (00/01/10/11) | Kupiec p | "
        "Independence p | Conditional coverage p |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for method, history in histories.items():
        hits = history["breach"]
        coverage = coverage_tests(hits, CONFIDENCE)
        transitions = "/".join(map(str, coverage.transitions))
        p_values = [
            "N/A" if test is None else f"{test.p_value:.4g}"
            for test in (coverage.kupiec, coverage.independence, coverage.conditional)
        ]
        lines.append(
            f"| {METHOD_LABELS[method]} | {hits.sum()} | {hits.mean():.2%} | "
            f"{transitions} | " + " | ".join(p_values) + " |"
        )
    lines.extend(["", f"![Realized portfolio loss, VaR, ES and breaches for {title}]({filename}.png)"])
    return lines


def _write_plot(title: str, histories: dict[str, pd.DataFrame], path: Path) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
        from matplotlib.figure import Figure
        from matplotlib.ticker import PercentFormatter
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise ValueError("install the market extra with pip install '.[market]'") from error

    figure = Figure(figsize=(10, 7.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True, sharey=True)
    figure.suptitle(f"{title} · one-session portfolio loss and 99% risk")
    figure.supylabel("Share of portfolio value")
    for ax, (method, history) in zip(axes, histories.items(), strict=True):
        ax.plot(history.index, history["loss"], color="#667387", linewidth=0.75,
                alpha=0.75, label="Realized loss")
        ax.plot(history.index, history["var"], color="#184eac", linewidth=1.25, label="VaR")
        ax.plot(history.index, history["es"], color="#137b69", linewidth=1.1,
                linestyle="--", label="ES")
        breaches = history.loc[history["breach"], "loss"]
        ax.scatter(breaches.index, breaches, color="#c12d3a", s=19, zorder=4, label="VaR breach")
        ax.set_title(METHOD_LABELS[method], loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.22)
    locator = AutoDateLocator(minticks=4, maxticks=9)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    figure.savefig(path, dpi=150)

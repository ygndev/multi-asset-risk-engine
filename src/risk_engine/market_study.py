from pathlib import Path

import pandas as pd

from .attribution import rolling_volatility_attribution
from .backtest import rolling_backtest
from .diagnostics import max_drawdown, var_exceedances
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
        asset_returns = price_returns(prices)
        returns = portfolio_returns(asset_returns, WEIGHTS)
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
        _write_model_validation(output, asset_returns, histories, provenance["prices_sha256"])


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


def _write_model_validation(
    output: Path,
    asset_returns: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    price_checksum: str,
) -> None:
    attribution = {
        method: rolling_volatility_attribution(
            asset_returns, WEIGHTS, histories[method], method, WINDOW, DECAY
        )
        for method in ("normal", "ewma")
    }
    tails = {method: var_exceedances(history) for method, history in histories.items()}
    for filename, frames in (
        ("volatility-attribution.csv", attribution),
        ("var-exceedances.csv", tails),
    ):
        pd.concat(frames, names=["method"]).to_csv(
            output / filename, date_format="%Y-%m-%d", lineterminator="\n"
        )

    report = [
        "# Portfolio model validation",
        "",
        f"IVV 50%, IEF 30%, IAU 20% in USD, rebalanced each session. "
        f"One-session forecasts use the preceding {WINDOW} returns, {CONFIDENCE:.0%} confidence "
        f"and EWMA decay {DECAY}. Price snapshot SHA-256 `{price_checksum}`.",
        "",
        "[Forecasts and VaR coverage tests](report.md) · "
        "[Daily volatility attribution](volatility-attribution.csv) · "
        "[Dated exceedances and run IDs](var-exceedances.csv)",
        "",
        "## Volatility attribution",
        "",
        "Portfolio volatility is `sqrt(wᵀΣw)`. Marginal volatility is the change per unit "
        "weight with other weights held fixed, `(Σw)ᵢ/σ`. The component `wᵢ(Σw)ᵢ/σ` is "
        "each asset's signed contribution to total volatility. A negative covariance with "
        "the portfolio can make a positive-weight asset's component negative. Components "
        "sum to portfolio volatility on each date with defined contributions. These "
        "estimates decompose volatility. VaR and ES remain portfolio-level forecasts.",
        "",
        "The Gaussian estimate uses centered sample covariance with `ddof=1`. EWMA "
        "uses zero-mean outer products and initializes each window with the first outer "
        "product. Zero or numerically unresolved projected variance leaves the contributions "
        "undefined. Blank CSV contributions and N/A table values denote undefined estimates.",
        "",
        "Asset cells show mean marginal / component volatility on defined forecast dates. "
        "The reconciliation error is the maximum absolute difference between the sum of "
        "components and the forecast's portfolio volatility, in return units.",
        "",
        "| Period | Model | Defined forecasts | Mean portfolio volatility | IVV | IEF | IAU | "
        "Max reconciliation error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for title, _, dates in PERIODS:
        for method, frame in attribution.items():
            defined = frame.loc[dates].dropna(subset=["component_volatility"])
            forecast_volatility = defined["portfolio_volatility"].groupby(level=0).first()
            means = defined.groupby("asset")[["marginal_volatility", "component_volatility"]].mean()
            assets = [
                f"{row.marginal_volatility:.3%} / {row.component_volatility:.3%}"
                if pd.notna(row.component_volatility) else "N/A"
                for row in means.reindex(WEIGHTS).itertuples()
            ]
            residual = defined["component_volatility"].groupby(level=0).sum() - forecast_volatility
            mean_volatility = "N/A" if defined.empty else f"{forecast_volatility.mean():.3%}"
            max_error = "N/A" if defined.empty else f"{residual.abs().max():.2e}"
            report.append(
                f"| {title} | {METHOD_LABELS[method]} | {len(forecast_volatility):,} | "
                f"{mean_volatility} | " + " | ".join(assets) + f" | {max_error} |"
            )
    report.extend([
        "",
        "## VaR and ES diagnostics",
        "",
        "VaR exceedance size is realized loss minus the corresponding VaR. Consecutive "
        "means adjacent trading sessions. Runs are identified over the full sequence and "
        "counted within each reporting window. Dates display the first longest run in a tie.",
        "",
        "The ES comparison pairs realized loss and forecast ES on each model's own strict "
        "VaR exceedance dates. The average paired gap is descriptive. It depends on the "
        "VaR threshold, and the dates can differ across models. Ties at the VaR threshold "
        "also affect selection. A formal ES backtest "
        "requires a statistical procedure with a stated null and treatment of VaR "
        "calibration and serial dependence. See "
        "[Bayer and Dimitriadis](https://arxiv.org/abs/1801.04112).",
        "",
        "Short stress windows have few expected VaR hits, and small tail samples make "
        "mean losses imprecise. The stress windows are subsets of the full "
        "backtest, and overlapping forecast windows can introduce dependence.",
    ])
    for title, _, dates in PERIODS:
        period = histories["historical"].loc[dates]
        drawdown = max_drawdown(-period["loss"], period["forecast_as_of"].iloc[0])
        peak_trough = (
            f" ({drawdown.peak_date.date()} to {drawdown.trough_date.date()})"
            if drawdown.peak_date is not None else ""
        )
        report.extend([
            "",
            f"### {title}",
            "",
            f"{len(period):,} forecasts per model, {len(period) * (1 - CONFIDENCE):.2f} "
            f"nominal expected exceedances. Maximum drawdown {drawdown.magnitude:.2%}"
            f"{peak_trough}. Wealth starts at one on the session before this window.",
            "",
            "| Model | Hits | Hit rate | Mean loss above VaR | Largest excess (date) | "
            "Runs of 2+ | Longest run (dates) |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ])
        es_rows = []
        for method, tail in tails.items():
            window_tail = tail.loc[dates]
            label = METHOD_LABELS[method]
            hits = len(window_tail)
            va_row = f"| {label} | {hits} | {hits / len(period):.2%} |"
            if window_tail.empty:
                report.append(f"{va_row} N/A | N/A | 0 | N/A |")
                es_rows.append(f"| {label} | 0 | N/A | N/A | N/A |")
                continue
            runs = window_tail.groupby("run_id", sort=False).size()
            longest = window_tail[window_tail["run_id"] == runs.idxmax()]
            worst = window_tail["excess_over_var"].idxmax()
            report.append(
                f"{va_row} {window_tail['excess_over_var'].mean():.3%} | "
                f"{window_tail.at[worst, 'excess_over_var']:.3%} ({worst.date()}) | "
                f"{(runs >= 2).sum()} | {len(longest)} ({longest.index[0].date()} to "
                f"{longest.index[-1].date()}) |"
            )
            es_rows.append(
                f"| {label} | {hits} | {window_tail['loss'].mean():.3%} | "
                f"{window_tail['es'].mean():.3%} | {window_tail['loss_minus_es'].mean():+.3%} |"
            )
        report.extend([
            "",
            "| Model | ES tail observations | Mean realized tail loss | "
            "Mean ES on those dates | Mean loss minus ES |",
            "| --- | ---: | ---: | ---: | ---: |",
            *es_rows,
        ])
    (output / "model-validation.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )


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

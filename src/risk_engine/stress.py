import json
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import estimate_covariance, volatility_attribution
from .diagnostics import Drawdown, compound_wealth, peak_to_trough
from .market import DECAY, STRESS_PERIODS, WEIGHTS, WINDOW, new_output_directory, read_market_snapshot
from .prices import portfolio_returns, portfolio_weights, price_returns

PORTFOLIOS = {
    "Baseline": WEIGHTS,
    "Treasury tilt": {"IVV": 0.30, "IEF": 0.50, "IAU": 0.20},
    "Equity tilt": {"IVV": 0.70, "IEF": 0.10, "IAU": 0.20},
}


@dataclass(frozen=True)
class StressReplay:
    path: pd.DataFrame
    drawdown: Drawdown
    recovery_date: pd.Timestamp | None
    recovery_sessions: int | None


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"duplicate scenario JSON field: {key}")
        fields[key] = value
    return fields


def parse_scenarios(config: bytes) -> pd.DataFrame:
    """Read one-day percentage shocks and return labeled decimal returns."""
    document = json.loads(config, object_pairs_hook=_unique_fields)
    if not isinstance(document, dict) or set(document) != {"unit", "scenarios"}:
        raise ValueError("scenario JSON requires the fields unit and scenarios")
    if document["unit"] != "percent":
        raise ValueError('scenario unit must be "percent" (-5 means a 5% loss)')
    scenarios = document["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a nonempty list")

    returns = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {"name", "shocks"}:
            raise ValueError("each scenario requires a name and shocks")
        name = scenario["name"]
        if not isinstance(name, str) or not name.strip() or len(name.splitlines()) != 1:
            raise ValueError("scenario names must be nonempty, single-line text")
        name = name.strip()
        if name in returns:
            raise ValueError(f"duplicate scenario name: {name}")
        shocks = scenario["shocks"]
        if not isinstance(shocks, dict) or set(shocks) != set(WEIGHTS):
            raise ValueError(f"{name}: shocks must specify IVV, IEF and IAU exactly once")
        row = []
        for asset in WEIGHTS:
            percent = shocks[asset]
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise ValueError(f"{name}: {asset} shock must be a numeric percent")
            try:
                shock = percent / 100
            except OverflowError:
                raise ValueError(f"{name}: {asset} shock exceeds the numeric range") from None
            if not isfinite(shock) or shock < -1:
                raise ValueError(f"{name}: {asset} shock must be finite and at least -100%")
            row.append(shock)
        returns[name] = row
    return pd.DataFrame.from_dict(returns, orient="index", columns=list(WEIGHTS))


def replay_period(returns: pd.Series, dates: slice) -> StressReplay:
    """Replay realized returns, with recovery followed through the final observed close."""
    index = returns.index
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError("stress replay requires chronological and unique session dates")
    period = returns.loc[dates]
    if period.empty or period.index[0] == index[0]:
        raise ValueError("stress replay requires period returns and a preceding session")
    first = index.get_loc(period.index[0])
    baseline = index[first - 1]
    wealth = compound_wealth(returns.iloc[first:], baseline)
    period_wealth = wealth.loc[:period.index[-1]]
    drawdown = peak_to_trough(period_wealth)
    path = pd.DataFrame({
        "portfolio_return": period,
        "cumulative_return": period_wealth.iloc[1:] - 1,
        "drawdown": (1 - period_wealth / period_wealth.cummax()).iloc[1:],
    }).rename_axis("date")
    if drawdown.trough_date is None:
        return StressReplay(path, drawdown, None, None)

    trough_index = wealth.index.get_loc(drawdown.trough_date)
    later = wealth.iloc[trough_index + 1:]
    recovered = later[later >= wealth.at[drawdown.peak_date]]
    if recovered.empty:
        return StressReplay(path, drawdown, None, len(later))
    recovery_date = recovered.index[0]
    sessions = wealth.index.get_loc(recovery_date) - trough_index
    return StressReplay(path, drawdown, recovery_date, sessions)


def pre_stress_covariances(
    returns: pd.DataFrame, first_session: pd.Timestamp
) -> tuple[pd.Timestamp, dict[str, np.ndarray]]:
    """Fit both covariance estimators on the same pre-event asset return window."""
    sample = returns.loc[returns.index < first_session].tail(WINDOW)
    if len(sample) < WINDOW:
        raise ValueError(f"need {WINDOW} returns before the first stress session")
    return sample.index[-1], {
        method: estimate_covariance(sample, method, DECAY) for method in ("normal", "ewma")
    }


def run_stress_test(snapshot: Path, scenario_file: Path, destination: Path) -> None:
    """Publish the historical replay, fixed-weight sensitivities, and hypothetical P&L."""
    with new_output_directory(destination) as output:
        config = scenario_file.read_bytes()
        shocks = parse_scenarios(config)
        prices, provenance = read_market_snapshot(snapshot)
        asset_returns = price_returns(prices)
        portfolios = {
            name: portfolio_returns(asset_returns, weights) for name, weights in PORTFOLIOS.items()
        }
        report = [
            "# Portfolio stress and scenario analysis",
            "",
            f"USD adjusted closes from {provenance['source']}. "
            f"Price snapshot SHA-256 `{provenance['prices_sha256']}`.",
            "",
            "| Portfolio | IVV | IEF | IAU |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {name} | {weights['IVV']:.0%} | {weights['IEF']:.0%} | {weights['IAU']:.0%} |"
                for name, weights in PORTFOLIOS.items()
            ],
            "",
            "The fixed, long-only weights are set before comparison. Each session starts at "
            "those weights. Returns are close to close, with zero trading costs and taxes.",
            "",
            "## Historical replay",
            "",
            "Observed asset returns drive the portfolio paths. Each window compounds from "
            "wealth of one at the preceding close. Positive cumulative loss means a "
            "decline. Recoveries refer to the peak preceding the window's maximum drawdown. "
            "The recovery clock counts subsequent trading sessions from the trough until "
            f"the peak is regained or the snapshot ends on {asset_returns.index[-1].date()}. "
            "Post-window recovery leaves the in-window outcomes unchanged.",
            "",
            "[Daily portfolio paths](historical-paths.csv) · "
            "[Cumulative return chart](stress-paths.png)",
            "",
            "| Period | Portfolio | Sessions | Cumulative loss | Max drawdown (peak → trough) | "
            "Worst session return | End drawdown | Recovery |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
        paths = {}
        sensitivity_rows = []
        for title, slug, dates in STRESS_PERIODS:
            first_session = asset_returns.loc[dates].index[0]
            as_of, covariances = pre_stress_covariances(asset_returns, first_session)
            for name, returns in portfolios.items():
                replay = replay_period(returns, dates)
                path = replay.path
                paths[slug, name] = path
                worst = path["portfolio_return"].idxmin()
                cumulative_loss = -path["cumulative_return"].iloc[-1]
                drawdown = f"{replay.drawdown.magnitude:.2%}"
                if replay.drawdown.peak_date is not None:
                    drawdown += (
                        f" ({replay.drawdown.peak_date.date()} → {replay.drawdown.trough_date.date()})"
                    )
                if replay.recovery_date is not None:
                    timing = "after window" if replay.recovery_date > path.index[-1] else "in window"
                    recovery = (
                        f"{replay.recovery_date.date()} ({replay.recovery_sessions} sessions, {timing})"
                    )
                elif replay.recovery_sessions is not None:
                    recovery = f"Open at snapshot end ({replay.recovery_sessions} sessions)"
                else:
                    recovery = "N/A"
                report.append(
                    f"| {title} | {name} | {len(path)} | {cumulative_loss:.2%} | {drawdown} | "
                    f"{path.at[worst, 'portfolio_return']:.2%} ({worst.date()}) | "
                    f"{path['drawdown'].iloc[-1]:.2%} | {recovery} |"
                )
                weights = portfolio_weights(asset_returns.columns, PORTFOLIOS[name])
                gaussian, ewma = (
                    volatility_attribution(covariances[method], weights).volatility
                    for method in ("normal", "ewma")
                )
                sensitivity_rows.append(
                    f"| {title} | {name} | {as_of.date()} | {gaussian:.3%} | {ewma:.3%} | "
                    f"{cumulative_loss:.2%} |"
                )
        report.extend([
            "",
            "## Weight sensitivity",
            "",
            f"One-session volatility estimates use the same {WINDOW} asset returns ending "
            "at the covariance date, before the first stress return. Gaussian uses centered "
            f"sample covariance and EWMA uses uncentered returns with decay {DECAY}. "
            "Each covariance is shared across the three allocations. The cumulative loss "
            "is the separately realized replay outcome for each fixed allocation.",
            "",
            "| Period | Portfolio | Covariance as of | Gaussian volatility | EWMA volatility | Cumulative loss |",
            "| --- | --- | --- | ---: | ---: | ---: |",
            *sensitivity_rows,
            "",
            "## Hypothetical one-session shocks",
            "",
            "Input shocks are numeric percentages. P&L is the weighted sum of the three "
            "asset shocks and is shown as a percentage of starting portfolio value. These "
            "assumptions have no assigned probabilities or modeled follow-through.",
            "",
            f"[Scenario configuration](scenarios.json) · SHA-256 `{sha256(config).hexdigest()}`.",
            "",
            "| Scenario | IVV shock | IEF shock | IAU shock | "
            + " | ".join(portfolios) + " |",
            "| --- | ---: | ---: | ---: |" + " ---: |" * len(portfolios),
        ])
        pnl = pd.DataFrame({
            name: portfolio_returns(shocks, weights) for name, weights in PORTFOLIOS.items()
        })
        for name, shock in shocks.iterrows():
            values = [*shock, *pnl.loc[name]]
            label = escape(name).replace("|", "&#124;")
            report.append(f"| {label} | " + " | ".join(f"{100 * value:+.6g}%" for value in values) + " |")
        report.extend([
            "",
            "Historical outcomes depend on the chosen episodes and the adjusted-close "
            "snapshot. The simple portfolio ignores execution costs, intraday paths and "
            "liquidity effects. Single-session shocks describe simultaneous changes "
            "and provide no multi-session recovery path.",
            "",
        ])
        pd.concat(paths, names=["period", "portfolio"]).to_csv(
            output / "historical-paths.csv", date_format="%Y-%m-%d", lineterminator="\n"
        )
        (output / "scenarios.json").write_bytes(config)
        (output / "stress-report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        _plot_replays(paths, output / "stress-paths.png")


def _plot_replays(paths: dict[tuple[str, str], pd.DataFrame], output: Path) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
        from matplotlib.figure import Figure
        from matplotlib.ticker import PercentFormatter
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        raise ValueError("install the market extra with pip install '.[market]'") from error

    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharey=True)
    figure.suptitle("Historical stress replay · cumulative portfolio return")
    figure.supylabel("Change since the preceding close")
    colors = ("#184eac", "#137b69", "#a64b25")
    for ax, (title, slug, _) in zip(axes, STRESS_PERIODS, strict=True):
        for name, color in zip(PORTFOLIOS, colors, strict=True):
            trace = paths[slug, name]["cumulative_return"]
            ax.plot(trace.index, trace, color=color, label=name, linewidth=1.5)
        dates = paths[slug, "Baseline"].index
        padding = pd.Timedelta(hours=12)
        ax.set_xlim(dates[0] - padding, dates[-1] + padding)
        ax.axhline(0, color="#536071", linewidth=0.8, linestyle="--")
        ax.set_title(title, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.22)
        locator = AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, show_offset=False))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    figure.savefig(output, dpi=150)

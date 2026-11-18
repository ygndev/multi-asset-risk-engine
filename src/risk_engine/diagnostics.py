from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Drawdown:
    magnitude: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None


def var_exceedances(history: pd.DataFrame) -> pd.DataFrame:
    """Pair strict VaR exceedances with their forecasts and trading-session runs."""
    hits = history["loss"] > history["var"]
    run_ids = (hits & ~hits.shift(fill_value=False)).cumsum()
    tail = history.loc[hits, ["forecast_as_of", "loss", "var", "es"]]
    return tail.assign(
        excess_over_var=tail["loss"] - tail["var"],
        loss_minus_es=tail["loss"] - tail["es"],
        run_id=run_ids.loc[hits],
    )


def max_drawdown(returns: pd.Series, baseline_date: pd.Timestamp) -> Drawdown:
    """Measure a period's peak-to-trough loss from wealth of one before its first return."""
    dates = returns.index
    if (
        returns.empty
        or not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not baseline_date < dates[0]
    ):
        raise ValueError("drawdown requires ordered return dates after the baseline date")
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("drawdown requires finite returns of at least -100%")
    wealth = pd.concat([pd.Series([1.0], index=[baseline_date]), (1 + returns).cumprod()])
    if not np.isfinite(wealth).all():
        raise ValueError("compounded wealth exceeded the supported numeric range")
    declines = 1 - wealth / wealth.cummax()
    trough = declines.idxmax()
    magnitude = float(declines.at[trough])
    if magnitude == 0:
        return Drawdown(0.0, None, None)
    prefix = wealth.loc[:trough]
    peak = prefix[prefix == prefix.max()].index[-1]
    return Drawdown(magnitude, peak, trough)

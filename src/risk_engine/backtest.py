import numpy as np
import pandas as pd

from .models import estimate_risk


def rolling_backtest(
    returns: pd.Series,
    method: str,
    window: int,
    confidence: float = 0.99,
    decay: float = 0.94,
) -> pd.DataFrame:
    """Forecast each target date from exactly the preceding `window` returns."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("backtest window must be an integer of at least two")
    dates = returns.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("return dates must be unique, valid, and increasing")
    if len(returns) <= window:
        raise ValueError(f"need at least {window + 1} returns for the backtest")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("backtest returns must be finite")

    forecasts = [
        estimate_risk(values[target - window : target], method, confidence, decay)
        for target in range(window, len(values))
    ]
    history = pd.DataFrame(
        {
            "forecast_as_of": dates[window - 1:-1],
            "loss": -values[window:],
            "var": [forecast.var for forecast in forecasts],
            "es": [forecast.es for forecast in forecasts],
            "volatility": [forecast.volatility for forecast in forecasts],
        },
        index=dates[window:].rename("target_date"),
    )
    return history.assign(breach=history["loss"] > history["var"])

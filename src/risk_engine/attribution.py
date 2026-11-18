from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isclose, sqrt

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .prices import portfolio_weights


@dataclass(frozen=True)
class VolatilityAttribution:
    volatility: float
    marginal: NDArray[np.float64] | None
    component: NDArray[np.float64] | None


def estimate_covariance(
    returns: ArrayLike, method: str, decay: float = 0.94
) -> NDArray[np.float64]:
    """Estimate covariance using rows of chronological asset returns."""
    sample = np.asarray(returns, dtype=float)
    if sample.ndim != 2 or len(sample) < 2 or not sample.shape[1]:
        raise ValueError("covariance requires two or more return rows and at least one asset")
    if not np.isfinite(sample).all():
        raise ValueError("asset returns must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        if method == "normal":
            centered = sample - sample.mean(axis=0)
            normalized = centered / sqrt(len(sample) - 1)
        elif method == "ewma":
            if not 0 < decay < 1:
                raise ValueError("EWMA decay must be between zero and one (exclusive)")
            exponents = np.arange(len(sample) - 1, -1, -1)
            time_weights = (1 - decay) * decay**exponents
            # Seeding with the first outer product gives it the full remaining weight.
            time_weights[0] = decay**exponents[0]
            normalized = sample * np.sqrt(time_weights[:, None])
        else:
            raise ValueError(f"unknown covariance method: {method}")
        covariance = normalized.T @ normalized
    if not np.isfinite(covariance).all():
        raise ValueError("covariance exceeded the supported numeric range")
    return covariance


def volatility_attribution(covariance: ArrayLike, weights: ArrayLike) -> VolatilityAttribution:
    """Allocate one-period volatility via the Euler derivative at the given weights."""
    matrix = np.asarray(covariance, dtype=float)
    allocations = np.asarray(weights, dtype=float)
    if allocations.ndim != 1 or not len(allocations) or matrix.shape != (
        len(allocations), len(allocations)
    ):
        raise ValueError("covariance must be square with one row per weight")
    if not np.isfinite(matrix).all() or not np.isfinite(allocations).all():
        raise ValueError("covariance and weights must be finite")

    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return VolatilityAttribution(0.0, None, None)
    scaled = matrix / scale
    tolerance = 32 * len(allocations) * np.finfo(float).eps
    if np.max(np.abs(scaled - scaled.T)) > tolerance:
        raise ValueError("covariance must be symmetric")
    scaled = (scaled + scaled.T) / 2
    if np.linalg.eigvalsh(scaled)[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")

    weight_scale = float(np.max(np.abs(allocations)))
    if weight_scale == 0:
        return VolatilityAttribution(0.0, None, None)
    unit_weights = allocations / weight_scale
    projected = scaled @ unit_weights
    variance = float(unit_weights @ projected)
    unsigned_weights = np.abs(unit_weights)
    # The unsigned quadratic form bounds the scale of cancellation error in the variance.
    rounding_bound = tolerance * (unsigned_weights @ np.abs(scaled) @ unsigned_weights)
    if variance < -rounding_bound:
        raise ValueError("portfolio variance is negative")
    root = sqrt(max(0.0, variance))
    volatility = weight_scale * (sqrt(scale) * root)
    if not np.isfinite(volatility):
        raise ValueError("portfolio volatility exceeded the supported numeric range")
    if variance <= rounding_bound or volatility == 0:
        return VolatilityAttribution(volatility, None, None)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        marginal = projected * (sqrt(scale) / root)
        component = allocations * marginal
    if not np.isfinite(marginal).all() or not np.isfinite(component).all():
        raise ValueError("volatility contributions exceeded the supported numeric range")
    # Normalize the signed sum to avoid overflow when large components cancel.
    largest = max(volatility, float(np.max(np.abs(component))))
    if abs(fsum(component / largest) - volatility / largest) > tolerance:
        raise ValueError("component volatility contributions do not sum to portfolio volatility")
    return VolatilityAttribution(volatility, marginal, component)


def rolling_volatility_attribution(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    history: pd.DataFrame,
    method: str,
    window: int,
    decay: float = 0.94,
) -> pd.DataFrame:
    """Attribute each forecast's volatility using the same pre-target return window."""
    dates = asset_returns.index
    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window < 2
        or len(asset_returns) <= window
        or not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not history.index.equals(dates[window:])
        or not pd.DatetimeIndex(history["forecast_as_of"]).equals(dates[window - 1:-1])
    ):
        raise ValueError("attribution requires aligned forecasts and chronological asset returns")
    allocations = portfolio_weights(asset_returns.columns, weights)
    values = asset_returns.to_numpy(dtype=float)
    rows = []
    for target, (as_of, volatility) in enumerate(
        history[["forecast_as_of", "volatility"]].itertuples(index=False, name=None), window
    ):
        covariance = estimate_covariance(values[target - window:target], method, decay)
        attribution = volatility_attribution(covariance, allocations)
        if attribution.component is not None and not isclose(
            attribution.volatility, volatility, rel_tol=1e-8, abs_tol=1e-14
        ):
            raise ValueError(f"covariance and forecast volatility disagree on {dates[target].date()}")
        for i, asset in enumerate(asset_returns.columns):
            rows.append((
                dates[target],
                as_of,
                asset,
                allocations[i],
                volatility,
                attribution.marginal[i] if attribution.marginal is not None else np.nan,
                attribution.component[i] if attribution.component is not None else np.nan,
            ))
    return pd.DataFrame(
        rows, columns=[
            "target_date", "forecast_as_of", "asset", "weight", "portfolio_volatility",
            "marginal_volatility", "component_volatility",
        ]
    ).set_index("target_date")

import csv
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def read_prices(path: str | Path, *, sort: bool = True) -> pd.DataFrame:
    """Read a wide, daily adjusted-close CSV, sorting dates by default."""
    dates = []
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(header) < 2 or header[0] != "date":
            raise ValueError("price CSV requires a 'date' column followed by asset symbols")
        symbols = [name.strip() for name in header[1:]]
        if "" in symbols or len(set(symbols)) != len(symbols) or "date" in symbols:
            raise ValueError("asset symbols must be nonempty, unique, and different from 'date'")

        for row in reader:
            line = reader.line_num
            if len(row) != len(header):
                raise ValueError(f"line {line}: expected {len(header)} fields, got {len(row)}")
            try:
                observation = date.fromisoformat(row[0])
                if observation.isoformat() != row[0]:
                    raise ValueError
            except ValueError:
                raise ValueError(f"line {line}: expected a date in YYYY-MM-DD format") from None

            closes = []
            for symbol, field in zip(symbols, row[1:]):
                try:
                    close = float(field)
                except ValueError:
                    close = np.nan
                if not np.isfinite(close) or close <= 0:
                    raise ValueError(f"line {line}: {symbol} requires a finite, positive price")
                closes.append(close)
            dates.append(observation)
            rows.append(closes)

    if len(rows) < 2:
        raise ValueError("at least two price observations are required")
    index = pd.DatetimeIndex(dates, name="date")
    if not index.is_unique:
        raise ValueError(f"duplicate price date: {index[index.duplicated()][0].date()}")
    prices = pd.DataFrame(rows, index=index, columns=symbols)
    return prices.sort_index() if sort else prices


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple returns across successive common price observations."""
    dates = prices.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError("price dates must be unique, valid, and increasing")
    if len(prices) < 2 or prices.shape[1] == 0 or not prices.columns.is_unique:
        raise ValueError("at least two price rows and unique asset columns are required")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices must be finite and positive")

    with np.errstate(over="ignore", under="ignore"):
        returns = values[1:] / values[:-1] - 1
    if not np.isfinite(returns).all():
        raise ValueError("price ratios exceeded the supported numeric range")
    return pd.DataFrame(returns, index=dates[1:], columns=prices.columns)


def portfolio_returns(
    returns: pd.DataFrame, weights: Mapping[str, float]
) -> pd.Series:
    """Aggregate asset returns for a fully invested, fixed-weight long-only portfolio."""
    if not returns.columns.is_unique or set(returns.columns) != set(weights):
        raise ValueError("weight symbols must match the asset columns exactly")
    allocations = np.array([weights[symbol] for symbol in returns.columns], dtype=float)
    if not np.isfinite(allocations).all() or np.any(allocations < 0):
        raise ValueError("portfolio weights must be finite and nonnegative")
    if not np.isclose(allocations.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("portfolio weights must sum to one")
    return returns.dot(allocations).rename("portfolio_return")

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .prices import read_prices

WEIGHTS = {"IVV": 0.50, "IEF": 0.30, "IAU": 0.20}
WINDOW = 250
DECAY = 0.94
STRESS_PERIODS = (
    ("March 2020", "march-2020", slice("2020-03-01", "2020-03-31")),
    ("Calendar year 2022", "2022", slice("2022-01-01", "2022-12-31")),
)
START = "2010-01-01"
LAST_DATE = "2024-12-31"
END_EXCLUSIVE = "2025-01-01"
CALENDAR = "XNYS"
SOURCE = "Yahoo Finance via yfinance"
SNAPSHOT_SPEC = {
    "source": SOURCE,
    "symbols": list(WEIGHTS),
    "currency": "USD",
    "start": START,
    "end_exclusive": END_EXCLUSIVE,
    "interval": "1d",
    "auto_adjust": True,
    "price_field": "Close",
    "adjustment": "yfinance auto-adjusted close for splits and distributions",
    "calendar": CALENDAR,
}


@contextmanager
def new_output_directory(destination: Path):
    """Publish a complete set of files without replacing an existing output."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".risk-engine-") as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        yield staged
        staged.rename(destination)


def expected_sessions() -> pd.DatetimeIndex:
    try:
        from exchange_calendars import get_calendar
    except ModuleNotFoundError as error:
        if error.name != "exchange_calendars":
            raise
        raise ValueError("install the market extra with pip install '.[market]'") from error
    return get_calendar(CALENDAR, start=START, end=LAST_DATE).sessions


def validate_market_prices(prices: pd.DataFrame, sessions: pd.DatetimeIndex) -> None:
    """Require a positive adjusted close for every symbol on every expected session."""
    if not prices.columns.is_unique or set(prices.columns) != set(WEIGHTS):
        raise ValueError(f"adjusted close columns must be exactly {', '.join(WEIGHTS)}")
    dates = prices.index
    if not isinstance(dates, pd.DatetimeIndex) or dates.tz is not None or dates.hasnans:
        raise ValueError("market dates must be valid, timezone-naive session dates")
    if not dates.is_unique:
        raise ValueError(f"duplicate market date: {dates[dates.duplicated()][0].date()}")
    if not dates.is_monotonic_increasing or not dates.equals(dates.normalize()):
        raise ValueError("market dates must be in chronological order with no time component")

    missing = sessions.difference(dates)
    unexpected = dates.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"{CALENDAR} calendar mismatch: {len(missing)} missing sessions "
            f"({', '.join(missing[:3].strftime('%Y-%m-%d')) or 'none'}); "
            f"{len(unexpected)} unexpected dates "
            f"({', '.join(unexpected[:3].strftime('%Y-%m-%d')) or 'none'})"
        )
    values = prices.to_numpy(dtype=float)
    rows, columns = np.where(~np.isfinite(values) | (values <= 0))
    if len(rows):
        raise ValueError(
            f"{prices.columns[columns[0]]} on {dates[rows[0]].date()} "
            "requires a finite, positive adjusted close"
        )


def fetch_market_prices(destination: Path) -> None:
    """Download and save the fixed study dataset and its provenance."""
    with new_output_directory(destination) as staged:
        try:
            import yfinance as yf
            from yfinance.exceptions import YFException
        except ModuleNotFoundError as error:
            if error.name != "yfinance":
                raise
            raise ValueError("install the market extra with pip install '.[market]'") from error

        sessions = expected_sessions()
        try:
            downloaded = yf.download(
                list(WEIGHTS),
                start=START,
                end=END_EXCLUSIVE,
                interval="1d",
                auto_adjust=True,
                keepna=True,
                ignore_tz=True,
                group_by="column",
                multi_level_index=True,
                threads=False,
                progress=False,
                timeout=15,
            )
        except (OSError, YFException) as error:
            raise ValueError(f"Yahoo Finance download failed: {error}") from error
        retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
        if downloaded is None or downloaded.empty:
            raise ValueError("Yahoo Finance returned no data for IVV, IEF and IAU")
        if (
            not isinstance(downloaded.columns, pd.MultiIndex)
            or downloaded.columns.nlevels != 2
            or "Close" not in downloaded.columns.get_level_values(0)
        ):
            raise ValueError("Yahoo Finance response is missing the adjusted Close field")
        prices = downloaded["Close"]
        validate_market_prices(prices, sessions)

        csv_path = staged / "prices.csv"
        prices.loc[:, list(WEIGHTS)].to_csv(
            csv_path, index_label="date", date_format="%Y-%m-%d", lineterminator="\n"
        )
        provenance = {
            **SNAPSHOT_SPEC,
            "retrieved_at_utc": retrieved_at,
            "yfinance_version": version("yfinance"),
            "exchange_calendars_version": version("exchange-calendars"),
            "first_session": prices.index[0].date().isoformat(),
            "last_session": prices.index[-1].date().isoformat(),
            "observations": len(prices),
            "prices_sha256": sha256(csv_path.read_bytes()).hexdigest(),
        }
        (staged / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


def read_market_snapshot(directory: Path) -> tuple[pd.DataFrame, dict]:
    """Load and verify a snapshot created by the market fetch command."""
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or any(
        provenance.get(field) != expected for field, expected in SNAPSHOT_SPEC.items()
    ):
        raise ValueError("provenance does not match the fixed market study dataset")
    for field in ("retrieved_at_utc", "yfinance_version", "exchange_calendars_version"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"snapshot provenance is missing {field}")

    csv_path = directory / "prices.csv"
    if sha256(csv_path.read_bytes()).hexdigest() != provenance.get("prices_sha256"):
        raise ValueError("prices.csv SHA-256 does not match provenance.json")
    prices = read_prices(csv_path, sort=False)
    validate_market_prices(prices, expected_sessions())
    if (
        provenance.get("observations") != len(prices)
        or provenance.get("first_session") != prices.index[0].date().isoformat()
        or provenance.get("last_session") != prices.index[-1].date().isoformat()
    ):
        raise ValueError("snapshot observation count or date range does not match provenance")
    return prices, provenance

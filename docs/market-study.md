# Market study

## Reproduce the study

Install the optional market dependencies to run the full offline test suite and
reproduce the historical reports. Fetch once if you need a snapshot.

```sh
python -m pip install ".[market,test]"
python -m pytest -q
risk-engine fetch artifacts/market-data
risk-engine market-study artifacts/market-data artifacts/market-study
```

`fetch` requests daily USD prices for IVV, IEF, and IAU from Yahoo Finance via
[yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
It requests January 1, 2010 through December 31, 2024 using an exclusive API
end date of January 1, 2025. With `auto_adjust=True`, the returned `Close` field
accounts for splits and distributions. The study uses the XNYS regular-session
calendar as its common U.S. equity session reference. A missing or extra
session, a duplicate or out-of-order date, or an invalid price stops the fetch.
Download failures stop the command with an error.

The snapshot contains `prices.csv` in the same format accepted by `risk-engine
run` and `provenance.json` with the provider, request settings, retrieval time in
UTC, yfinance and calendar versions, observation range, and CSV SHA-256. Reuse
both files to replay the same observations. A fresh download can reflect provider
revisions. The backtest uses one retrospective data snapshot, so estimates are
chronological within that snapshot and inherit its vendor adjustments. Both
commands take new output directories, so previous runs can be kept side by side.
Keep downloaded data and generated market artifacts local under the ignored
`artifacts/` directory. Verify the provider's terms before any redistribution.
[yfinance data-use notice](https://ranaroussi.github.io/yfinance/)

The study fixes the weights at 50% IVV, 30% IEF, and 20% IAU. It uses a
250-return rolling window, 99% confidence, and EWMA decay of 0.94. It fits each
model along the full history using information through the preceding session,
then selects the March 2020 and calendar-year 2022 forecasts by target date.
The output includes `report.md` with nominal and observed VaR breaches,
Kupiec and Christoffersen p-values, and realized loss summaries. `forecasts.csv`
contains daily realized losses, VaR, ES, and breach flags. The PNGs show the
loss and forecasts for all three models over each period.

At 99% confidence, March 2020 has 0.22 and 2022 has 2.51 nominal expected
breaches per model. Even the full backtest has only about 0.35 expected
consecutive breach pairs under independent 1% hits. Small counts and sparse
transitions limit the precision of chi-square p-values, including Kupiec
coverage in short samples. Treat stress-window diagnostics as descriptive. The coverage tests evaluate VaR and
the additional model validation report describes ES on the observed tail dates.

## Model validation outputs

The same `market-study` command writes `model-validation.md` and two additional
CSVs. To reuse a previously fetched snapshot, run the study directly with a
new output directory.

```sh
risk-engine market-study artifacts/market-data artifacts/model-validation
```

- `volatility-attribution.csv` uses the same pre-target windows as the Gaussian
  and EWMA forecasts. For portfolio volatility `σ = sqrt(wᵀΣw)`, the marginal
  contribution is `(Σw)ᵢ / σ` and the component is `wᵢ(Σw)ᵢ / σ`. The Gaussian
  covariance is the centered sample covariance with `ddof=1`. EWMA weights
  uncentered outer products, with the first outer product as its initial
  covariance. The signed components sum to portfolio volatility. Negative
  components can occur when an asset has negative covariance with the
  portfolio, even at a positive weight. At zero or numerically unresolved
  variance, marginal and component values are undefined and appear as blanks.
  These allocations are in volatility units. [Tasche](https://arxiv.org/abs/0708.2542)
- `var-exceedances.csv` lists dates, realized loss, VaR, ES, the loss minus
  each forecast, and run IDs for consecutive VaR exceedances. The report
  summarizes frequency and magnitude and identifies maximum compounded
  drawdown for each period. The peak search starts at wealth one on the
  session immediately before the period's first forecast target.
- The ES table pairs realized tail losses with the ES forecast on the same
  dates. Its sample size counts strict VaR exceedances and can differ between
  models. The paired mean gap is descriptive and depends on the VaR threshold
  and tail sample. A formal ES backtest requires a statistical procedure with
  a specified null and assumptions about calibration and dependence.
  [Bayer and Dimitriadis](https://arxiv.org/abs/1801.04112)

## Reference snapshot

The reference run used 3,774 common price sessions from 2010-01-04 through
2024-12-31. The snapshot was retrieved on 2026-11-18 at 07:13:40 UTC
using yfinance 1.8.0 and exchange-calendars 4.13.3. The price CSV SHA-256 is
`40ad4c8e73ef6cc4a3f50458377335648ecec6179f2270d2ddac1b529a13940b`.
The price snapshot, daily forecasts, and market-derived report files and
figures are generated locally. Check provider permissions before
redistributing those files.

See the [README](../README.md#market-study-at-a-glance) for the aggregate
reference results and the [methodology](methodology.md) for the estimator
definitions. The study describes one historical sample. Formal ES inference,
trading performance, and regulatory compliance are outside its scope.

## Custom price CSV

To analyze your own prices, pass the CSV path and the portfolio weights.

```sh
risk-engine run prices.csv --weights EQ=0.50 BOND_ETF=0.30 GOLD_ETF=0.20 --window 250 --confidence 0.99 --decay 0.94
```

Use a UTF-8 CSV with one row per common observation date and one column per asset.

```csv
date,EQ,BOND_ETF,GOLD_ETF
2024-01-02,100.00,100.00,100.00
2024-01-03,101.10,99.95,100.20
```

The rows illustrate the schema. A backtest with window `W` needs at least
`W + 2` price rows. Dates must be unique and written as `YYYY-MM-DD`. The loader
sorts rows and rejects missing, non-finite, and non-positive prices. Supply
adjusted closes in a common currency on a prealigned calendar. Each pair of
successive rows defines one return interval. Verify the input calendar if rows
are absent, since the loader does not infer exchange sessions or time zones.

## Reproduction environment

The reference reports were verified on Windows 11 x64 with CPython 3.12.14,
NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, exchange-calendars 4.13.3,
Matplotlib 3.11.2, and yfinance 1.8.0. Record your Python version and package
versions with `python -VV` and `python -m pip list --format=freeze`. Renderer,
font, and numeric library differences can affect bytes across environments.
With this stack, exchange-calendars emits four pandas deprecation warnings
when constructing the XNYS calendar.

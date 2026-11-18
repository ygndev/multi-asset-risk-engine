# Multi-Asset Risk Engine & Model Validation

A Python research project for one-session portfolio VaR and Expected Shortfall (ES),
out-of-sample VaR validation, volatility attribution, and stress replay. The
[synthetic report](examples/synthetic-risk-report.md) runs without market data.

The [price loader](src/risk_engine/prices.py) computes aligned asset and portfolio
returns. The [estimators](src/risk_engine/models.py) and
[rolling backtest](src/risk_engine/backtest.py) forecast loss from earlier returns.
[Market ingestion](src/risk_engine/market.py) saves and checks a price snapshot.
[Market validation](src/risk_engine/market_study.py) and
[stress analysis](src/risk_engine/stress.py) reuse that snapshot for their reports.

## Run the project

Requires Python 3.11 or newer. The CI matrix covers Python 3.11–3.15 on Linux.
Create a virtual environment from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell. Install the package, run the
tests, and view the demo.

```sh
python -m pip install ".[test]"
python -m pytest -q
risk-engine demo
```

To save the demo report as UTF-8 in PowerShell, run the following from the
repository root.

```powershell
$report = risk-engine demo
if ($LASTEXITCODE -eq 0) {
    [IO.File]::WriteAllLines("$PWD/examples/synthetic-risk-report.md", [string[]]$report)
}
```

In a POSIX shell, use redirection.

```sh
risk-engine demo > examples/synthetic-risk-report.md
```

`python -m risk_engine` is equivalent to `risk-engine`. The default run uses a
250-return window, 99% confidence, and EWMA decay of 0.94. The demo uses a fixed
PCG64 seed of 42 to generate 1,500 correlated log-return shocks across synthetic
equity, bond ETF, and gold ETF series. It adds two fixed high-volatility periods
and compounds the shocks into prices on a weekday calendar. The included
example was generated with Python 3.12.14, NumPy 2.5.5, pandas 3.1.1, and SciPy
1.18.1.

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

## Reproduce the market study

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
coverage in short samples. Treat
stress-window diagnostics as descriptive. The coverage tests evaluate VaR and
the additional model validation report describes ES on the observed tail dates.

### Model validation outputs

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

### Stress testing and scenarios

Use the saved market snapshot and a JSON scenario file. The command also
copies the exact scenario file into the report directory for replay.

```sh
risk-engine stress-test artifacts/market-data examples/stress-scenarios.json artifacts/stress-test
```

The [example scenarios](examples/stress-scenarios.json) use `"unit": "percent"`.
Give each scenario a unique name and numeric shocks for IVV, IEF, and IAU.
`-5` means a 5% price loss and `-0.05` means a 0.05% loss. The one-session
portfolio P&L is the sum of each shock times the asset's starting weight.
The shocks are assumptions with no assigned probability.

`stress-report.md` replays the observed March 2020 and calendar-year 2022
close-to-close returns with daily rebalancing. It reports signed cumulative
loss, maximum drawdown, worst session, and how long it took to regain the
peak preceding the largest drawdown. Wealth starts at one at the close
before each window. The historical outcome stops at the window end, and
recovery is followed separately through the snapshot's last session. Recovery
allows a relative rounding tolerance of `1e-12` at the prior wealth peak.
The daily paths and a chart accompany the report.

Sensitivity compares baseline weights (50/30/20), a Treasury tilt
(30/50/20), and an equity tilt (70/10/20) in IVV/IEF/IAU order. Each
allocation is fixed, with the gold weight held constant. Each period
uses the same Gaussian and EWMA covariances across allocations, estimated
from the 250 sessions preceding the first replayed return. Those volatility
estimates use only information available before the episode. The replay is
specific to the chosen dates, with no trading costs, taxes, or intraday
price path.

## Risk and validation conventions

- **Portfolio.** Simple asset returns are combined by label as
  `r_p = Σ w_i r_i`. Weights are long-only, sum to one, and are reset at the start
  of each interval. Loss is `L = -r_p`. Forecasts are fractions of portfolio
  value and may be negative for a sufficiently favorable estimated distribution.
- **Historical.** VaR is the inverse empirical CDF at confidence `c`.
  ES is the average quantile over the upper `1 - c` probability mass. When the
  required tail mass includes a fraction of the VaR observation, that fraction
  is included at the VaR value. For losses 1, 2, 3, and 4 at 60% confidence, VaR
  is 3 and ES is `(4 + 0.6 × 3) / 1.6 = 3.625`. With `W` observations, the
  VaR estimate is the `k`th smallest loss, where `k = ceil(c × W)`. For
  independent losses from the same continuous distribution, the next loss
  strictly exceeds this estimate with probability `(W + 1 - k) / (W + 1)`.
  [Acerbi and Tasche](https://arxiv.org/abs/cond-mat/0104295)
- **Normal.** The loss mean is the negative sample return mean. The volatility
  is the sample standard deviation with `ddof=1`. For standard normal quantile
  `z = Φ⁻¹(c)`, loss VaR is `μ_L + σz` and ES is
  `μ_L + σφ(z)/(1 - c)`.
- **EWMA normal.** Uses the same normal formulas with a zero conditional mean.
  In each rolling window, initialize variance with the first squared portfolio
  return and apply `v_next = λv + (1 - λ)r_p²` through the last observed return.
  For fixed weights, this yields the same variance as applying the recurrence
  to asset outer products with the same initialization and projecting onto
  the portfolio weights. Higher `λ` gives older observations more influence.
  [RiskMetrics Technical Document, §5.2](https://zhiqiang.org/static/docs/riskmanage/RiskMetrics_-_Technical_Document.PDF)
- **Backtest.** A forecast for date `t` uses the previous `W` observations and
  is dated as of `t-1`. A breach occurs when realized loss is strictly greater
  than forecast VaR. The next-observation forecast in the report uses the last
  `W` observed returns.
- **Validation.** The Kupiec likelihood ratio tests whether the breach
  probability equals `1 - c` using all target dates. Christoffersen's independence
  test compares an i.i.d. hit rate with a first-order Markov alternative. His
  conditional coverage test compares the nominal i.i.d. hit rate with that same
  Markov alternative. Both Christoffersen tests condition on the first hit
  indicator, so their likelihoods use only transitions. The joint statistic
  equals independence plus the coverage statistic *on those transitions*.
  The report shows transition counts alongside asymptotic chi-square p-values
  (degrees of freedom 1, 1, and 2). An absent origin or destination state makes
  both transition tests unavailable in this implementation. [Kupiec](https://fedinprint.org/item/fedgfe/34596/original) · [Christoffersen](https://doi.org/10.2307/2527341)

## Findings and limits

The saved 2010–2024 study produced 3,523 forecasts per model. Historical, normal,
and EWMA-normal VaR had 49, 76, and 67 breaches, respectively, against 35.23
nominally expected. The fixed baseline lost 4.15% in March 2020 and 13.27% in
2022. The Treasury tilt lost less in both replays and took longer to regain
its prior peak after the 2022 trough. These are results for the stated portfolio,
weights and snapshot.

Loss-tail assumptions matter. Gaussian tails can understate extremes, and
the empirical 99% ES uses only 2.5 observations' worth of tail mass in a
250-return window. VaR counts, first-order breach dependence, and paired ES
tail summaries cover different aspects of model behavior. Formal ES inference,
trading performance, and regulatory compliance are outside the scope of the study.

### Synthetic reference

Across 1,250 forecasts per model, the historical, normal, and EWMA normal
models recorded 20, 24, and 15 breaches, respectively. The nominal rate of 1%
implies 12.5 expected breaches. At 99% confidence with a 250-return window,
historical VaR selects the third-largest loss. Under independent losses from
the same continuous distribution, this threshold has a predictive breach
rate of `3 / 251` (1.195%, about 14.94 expected breaches over 1,250 forecasts).
The backtest still uses the nominal 1% target for all models.

EWMA normal was closest to the nominal target in this simulation. The two
normal models differ in both their volatility and mean estimates. The
constant-volatility normal model uses the sample mean, and EWMA normal
sets the mean to zero. EWMA normal's conditional coverage p-value was
0.6559, based on a sequence with zero consecutive breaches. Sparse transitions
limit the reliability of these asymptotic tests.

### Reproduction environment

The reference reports were verified on Windows 11 x64 with CPython 3.12.14,
NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, exchange-calendars 4.13.3,
Matplotlib 3.11.2, and yfinance 1.8.0. Record your Python version and package
versions with `python -VV` and `python -m pip list --format=freeze`. Renderer,
font, and numeric library differences can affect bytes across environments.
With this stack, exchange-calendars emits four pandas deprecation warnings
when constructing the XNYS calendar.

# Multi-Asset Risk Engine & Model Validation

A Python research project for one-period portfolio VaR, Expected Shortfall (ES), and
out-of-sample VaR validation. The [synthetic risk report](examples/synthetic-risk-report.md)
compares historical simulation, a normal model with sample volatility, and a normal
model with EWMA volatility on the same fixed-weight portfolio.

## Run the project

Requires Python 3.11 or newer. Create a virtual environment from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell. Install the package, run the
tests, and view the demo.

```sh
python -m pip install -e ".[test]"
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

## Reading the example

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

The likelihood-ratio tests assess VaR coverage and breach dependence. ES
calibration is outside this version's scope. With a 250-return window, the 99%
historical ES draws on 2.5 observations' worth of tail mass. The engine assumes
frictionless rebalancing and has a single observation horizon.

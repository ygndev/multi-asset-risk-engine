# Stress testing

Use the saved market snapshot and a JSON scenario file. The command also
copies the exact scenario file into the report directory for replay.

```sh
risk-engine stress-test artifacts/market-data examples/stress-scenarios.json artifacts/stress-test
```

The [example scenarios](../examples/stress-scenarios.json) use `"unit": "percent"`.
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

The Treasury tilt lost less in both reference replays and took longer than the
baseline to regain its prior peak after the 2022 trough. Use a verified
snapshot from the [market study](market-study.md) to reproduce the local chart
and report. The results depend on the chosen periods, allocations, and price
snapshot.

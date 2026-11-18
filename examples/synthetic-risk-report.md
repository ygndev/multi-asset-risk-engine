# Portfolio risk report

- Source: Synthetic prices (PCG64 seed 42; weekday calendar)
- Portfolio: EQ 50.00%, BOND_ETF 30.00%, GOLD_ETF 20.00%
- As of: 2023-10-03
- Horizon: one observation step
- Confidence: 0.99
- Lookback: 250 returns
- EWMA decay: 0.94

## Next-observation forecast

| Method | Loss VaR | Loss ES | Volatility |
| --- | ---: | ---: | ---: |
| Historical | 1.3519% | 1.4169% | 0.6322% |
| Normal | 1.4898% | 1.7041% | 0.6322% |
| EWMA normal | 1.5538% | 1.7801% | 0.6679% |

## Out-of-sample VaR backtest

2018-12-19 to 2023-10-03 (1,250 per model). Nominal breach rate: 1%.

| Method | Breaches | Breach rate | Transitions (00/01/10/11) | Kupiec LR / p | Christoffersen independence LR / p | Christoffersen conditional coverage LR / p |
| --- | ---: | ---: | --- | --- | --- | --- |
| Historical | 20 | 1.60% | 1213 / 17 / 16 / 3 | 3.8457 / 0.04987 | 9.1416 / 0.002499 | 12.9995 / 0.001504 |
| Normal | 24 | 1.92% | 1205 / 20 / 20 / 4 | 8.4188 / 0.003714 | 11.3368 / 0.0007599 | 19.7743 / 5.082e-05 |
| EWMA normal | 15 | 1.20% | 1219 / 15 / 15 / 0 | 0.4747 / 0.4908 | 0.3647 / 0.5459 | 0.8434 / 0.6559 |

A breach means realized loss strictly exceeds VaR. Sparse transitions make asymptotic p-values less reliable. N/A indicates an absent transition state. The likelihood-ratio tests assess VaR coverage and dependence.

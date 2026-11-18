import numpy as np
import pandas as pd

SEED = 42
WEIGHTS = {"EQ": 0.50, "BOND_ETF": 0.30, "GOLD_ETF": 0.20}


def synthetic_prices() -> pd.DataFrame:
    """Generate 1,500 correlated log-return observations with two high-volatility periods."""
    rng = np.random.Generator(np.random.PCG64(SEED))
    loadings = np.array(
        [
            [0.012, 0.0, 0.0],
            [-0.001, 0.004, 0.0],
            [0.002, 0.001, 0.009],
        ]
    )
    shocks = rng.standard_normal((1500, 3)) @ loadings.T
    shocks[420:550] *= 2.5
    shocks[1030:1160] *= 2.5
    log_prices = np.vstack([np.zeros(3), np.cumsum(shocks, axis=0)])
    dates = pd.bdate_range("2018-01-02", periods=len(log_prices), name="date")
    return pd.DataFrame(100 * np.exp(log_prices), index=dates, columns=list(WEIGHTS))

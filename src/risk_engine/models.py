from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

METHODS = ("historical", "normal", "ewma")


@dataclass(frozen=True)
class RiskEstimate:
    var: float
    es: float
    volatility: float


def validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one (exclusive)")


def estimate_risk(
    returns: ArrayLike,
    method: str,
    confidence: float = 0.99,
    decay: float = 0.94,
) -> RiskEstimate:
    """Forecast one-period loss VaR and ES from past portfolio returns only."""
    validate_confidence(confidence)
    tail = 1 - confidence
    if method not in METHODS:
        raise ValueError(f"unknown risk method: {method}")
    sample = np.asarray(returns, dtype=float)
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("risk estimation requires at least two finite portfolio returns")
    if method == "historical":
        losses = np.sort(-sample)
        size = len(losses)
        index = int(confidence * size)
        # Compare in probability space so rounding n * confidence cannot skip a CDF step.
        if index / size >= confidence:
            index -= 1
        var = losses[index]
        # Fill the remaining tail mass at VaR, including fractional observations.
        es = var + (losses[index + 1 :] - var).sum() / (size * tail)
        volatility = sample.std(ddof=1)
    else:
        if method == "ewma":
            if not 0 < decay < 1:
                raise ValueError("EWMA decay must be between zero and one (exclusive)")
            mean_loss = 0.0
            variance = sample[0] ** 2
            for realized in sample[1:]:
                variance = decay * variance + (1 - decay) * realized**2
            volatility = sqrt(variance)
        else:
            mean_loss = -sample.mean()
            volatility = sample.std(ddof=1)
        z = norm.ppf(confidence)
        var = mean_loss + volatility * z
        es = mean_loss + volatility * norm.pdf(z) / tail

    if not np.isfinite((var, es, volatility)).all():
        raise ValueError("risk estimate exceeded the supported numeric range")
    return RiskEstimate(float(var), float(es), float(volatility))

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlog1py, xlogy
from scipy.stats import chi2

from .models import validate_confidence


@dataclass(frozen=True)
class LikelihoodRatio:
    statistic: float
    p_value: float


@dataclass(frozen=True)
class CoverageTests:
    kupiec: LikelihoodRatio
    independence: LikelihoodRatio | None
    conditional: LikelihoodRatio | None
    transitions: tuple[int, int, int, int]  # 00, 01, 10, 11


def _bernoulli_loglik(zeros: int, ones: int, zero_probability: float) -> float:
    return float(xlogy(zeros, zero_probability) + xlog1py(ones, -zero_probability))


def _likelihood_ratio(null: float, alternative: float, df: int) -> LikelihoodRatio:
    statistic = max(0.0, 2 * (alternative - null))
    return LikelihoodRatio(statistic, float(chi2.sf(statistic, df)))


def coverage_tests(breaches: ArrayLike, confidence: float = 0.99) -> CoverageTests:
    """Test a chronological sequence of VaR breaches using Bernoulli likelihoods."""
    validate_confidence(confidence)
    hits = np.asarray(breaches)
    if hits.ndim != 1 or len(hits) == 0 or not np.isin(hits, [0, 1]).all():
        raise ValueError("breaches must be a nonempty sequence of binary indicators")
    hits = hits.astype(np.intp)
    ones = int(hits.sum())
    zeros = len(hits) - ones
    kupiec = _likelihood_ratio(
        _bernoulli_loglik(zeros, ones, confidence),
        _bernoulli_loglik(zeros, ones, zeros / len(hits)),
        df=1,
    )

    n00, n01, n10, n11 = map(
        int, np.bincount(2 * hits[:-1] + hits[1:], minlength=4)
    )
    transitions = (n00, n01, n10, n11)
    from_zero, from_one = n00 + n01, n10 + n11
    to_zero, to_one = n00 + n10, n01 + n11
    if min(from_zero, from_one, to_zero, to_one) == 0:
        return CoverageTests(kupiec, None, None, transitions)

    markov = _bernoulli_loglik(n00, n01, n00 / from_zero)
    markov += _bernoulli_loglik(n10, n11, n10 / from_one)
    # Both Christoffersen ratios condition on the first observed hit.
    iid = _bernoulli_loglik(to_zero, to_one, to_zero / (len(hits) - 1))
    fixed_rate = _bernoulli_loglik(to_zero, to_one, confidence)
    return CoverageTests(
        kupiec,
        _likelihood_ratio(iid, markov, df=1),
        _likelihood_ratio(fixed_rate, markov, df=2),
        transitions,
    )

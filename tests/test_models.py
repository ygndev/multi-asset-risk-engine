from math import pi, sqrt

import numpy as np
import pytest

from risk_engine.models import METHODS, estimate_risk


def test_historical_tail_includes_fractional_var_mass():
    risk = estimate_risk([-4, -1, -3, -2], "historical", confidence=0.60)
    assert risk.var == 3
    assert risk.es == pytest.approx(3.625)


def test_historical_ties_share_the_var_level():
    risk = estimate_risk([-2, -2, -4, -1], "historical", confidence=0.50)
    assert risk.var == 2
    assert risk.es == pytest.approx(3)


def test_historical_var_at_an_empirical_cdf_step():
    risk = estimate_risk(-np.arange(1, 101), "historical", confidence=0.56)
    assert risk.var == 56
    assert risk.es == pytest.approx(np.mean(np.arange(57, 101)))


def test_normal_model_uses_sample_mean_and_sample_volatility():
    risk = estimate_risk([-0.03, 0.01], "normal", confidence=0.95)
    sigma = sqrt(2) * 0.02
    assert risk.volatility == pytest.approx(sigma)
    assert risk.var == pytest.approx(0.01 + sigma * 1.644853626951)
    assert risk.es == pytest.approx(0.01 + sigma * 2.062712807507)


def test_ewma_updates_variance_in_chronological_order():
    risk = estimate_risk([0.10, -0.20, 0.30], "ewma", confidence=0.50, decay=0.5)
    sigma = sqrt(0.0575)
    assert risk.volatility == pytest.approx(sigma)
    assert risk.var == 0
    assert risk.es == pytest.approx(sigma * sqrt(2 / pi))


@pytest.mark.parametrize("method", METHODS)
def test_zero_return_risk(method):
    risk = estimate_risk([0.0, 0.0, 0.0], method)
    assert risk.var == risk.es == risk.volatility == 0


def test_risk_estimates_preserve_signed_losses():
    risk = estimate_risk([0.02, 0.02], "normal", confidence=0.99)
    assert risk.var == risk.es == -0.02


@pytest.mark.parametrize("confidence", [0, 1, float("nan")])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        estimate_risk([0.01, -0.01], "normal", confidence)


def test_nonfinite_return_and_invalid_decay_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        estimate_risk([0, np.inf], "historical")
    with pytest.raises(ValueError, match="decay"):
        estimate_risk([0.01, 0.02], "ewma", decay=1)

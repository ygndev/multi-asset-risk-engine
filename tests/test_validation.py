from math import erfc, exp, log, sqrt

import pytest

from risk_engine.validation import coverage_tests


def test_likelihood_ratios_against_hand_counted_probabilities():
    tests = coverage_tests([1, 1, 1, 0, 0, 0, 1, 0], confidence=0.75)
    assert tests.transitions == (2, 1, 2, 2)
    markov = (2 / 3) ** 2 * (1 / 3) * (1 / 2) ** 4
    iid = (4 / 7) ** 4 * (3 / 7) ** 3
    fixed_transitions = 0.75**4 * 0.25**3
    unconditional_lr = 2 * log(0.5**8 / (0.75**4 * 0.25**4))
    independence_lr = 2 * log(markov / iid)
    conditional_lr = 2 * log(markov / fixed_transitions)

    assert tests.kupiec.statistic == pytest.approx(unconditional_lr)
    assert tests.kupiec.p_value == pytest.approx(erfc(sqrt(unconditional_lr / 2)))
    assert tests.independence.statistic == pytest.approx(independence_lr)
    assert tests.independence.p_value == pytest.approx(erfc(sqrt(independence_lr / 2)))
    assert tests.conditional.statistic == pytest.approx(conditional_lr)
    assert tests.conditional.p_value == pytest.approx(exp(-conditional_lr / 2))


def test_clustering_changes_independence_but_not_kupiec():
    clustered = coverage_tests([0] * 80 + [1] * 20, confidence=0.80)
    dispersed = coverage_tests([0, 0, 0, 0, 1] * 20, confidence=0.80)
    assert clustered.kupiec == dispersed.kupiec
    assert clustered.kupiec.statistic == pytest.approx(0)
    assert clustered.independence.statistic > dispersed.independence.statistic


@pytest.mark.parametrize("hit", [0, 1])
def test_constant_breach_sequences_preserve_kupiec(hit):
    tests = coverage_tests([hit] * 10, confidence=0.95)
    assert tests.kupiec.statistic == pytest.approx(-20 * log(0.05 if hit else 0.95))
    assert tests.independence is None
    assert tests.conditional is None


def test_transitions_with_only_one_destination_are_unavailable():
    tests = coverage_tests([1, 0, 0], confidence=0.95)
    assert tests.transitions == (1, 0, 1, 0)
    assert tests.independence is None
    assert tests.conditional is None


@pytest.mark.parametrize("breaches", [[], [0, 2], [0, float("nan")]])
def test_nonbinary_or_empty_input_is_rejected(breaches):
    with pytest.raises(ValueError, match="binary"):
        coverage_tests(breaches)

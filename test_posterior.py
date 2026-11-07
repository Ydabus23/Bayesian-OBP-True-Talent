import math
import pytest
from posterior import likelihood_ratio, posterior_high, posterior_mean_obp


def binom_pmf(k, n, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def test_headline_result():
    assert round(posterior_high(42, 120), 2) == 0.66
    assert round(likelihood_ratio(42, 120), 2) == 1.95


def test_matches_direct_bayes_rule():
    k, n, lo, hi, prior = 42, 120, 0.300, 0.360, 0.5
    num = binom_pmf(k, n, hi) * prior
    den = num + binom_pmf(k, n, lo) * (1 - prior)
    assert posterior_high(k, n) == pytest.approx(num / den)


def test_unequal_prior_shifts_posterior():
    assert posterior_high(42, 120, prior_high=0.2) < posterior_high(42, 120) < posterior_high(42, 120, prior_high=0.8)


def test_more_evidence_same_rate_raises_confidence():
    vals = [posterior_high(round(0.35 * n), n) for n in (120, 240, 360, 480, 600)]
    assert vals == sorted(vals)


def test_identical_hypotheses_return_prior():
    assert posterior_high(42, 120, 0.33, 0.33, 0.4) == pytest.approx(0.4)


def test_large_sample_does_not_underflow():
    assert 0 < posterior_high(3500, 10000) <= 1


def test_posterior_mean_between_levels():
    assert 0.300 < posterior_mean_obp(42, 120) < 0.360


@pytest.mark.parametrize("args", [(-1, 10), (11, 10), (5, 0)])
def test_bad_counts(args):
    with pytest.raises(ValueError):
        posterior_high(*args)


def test_bad_probabilities():
    with pytest.raises(ValueError):
        posterior_high(5, 10, p_low=0.0)

"""Two-point Bayesian updating of a hitter's true-talent OBP.

Hypothesis: a batter's true talent is either `p_low` or `p_high`.
Evidence: `k` times on base in `n` plate appearances (binomial likelihood).
"""
import math


def _check(k, n, probs, prior_high):
    if not (isinstance(n, int) and isinstance(k, int) and 0 <= k <= n and n > 0):
        raise ValueError("need integers with 0 <= k <= n and n > 0")
    if any(not 0 < p < 1 for p in probs):
        raise ValueError("OBP values must be strictly between 0 and 1")
    if not 0 < prior_high < 1:
        raise ValueError("prior_high must be strictly between 0 and 1")


def likelihood_ratio(k, n, p_low=0.300, p_high=0.360):
    """P(data | p_high) / P(data | p_low). The binomial coefficient cancels."""
    _check(k, n, (p_low, p_high), 0.5)
    log_r = k * math.log(p_high / p_low) + (n - k) * math.log((1 - p_high) / (1 - p_low))
    return math.exp(log_r)


def posterior_high(k, n, p_low=0.300, p_high=0.360, prior_high=0.5):
    """P(true talent == p_high | k on base in n PA). Computed in log space."""
    _check(k, n, (p_low, p_high), prior_high)
    log_r = k * math.log(p_high / p_low) + (n - k) * math.log((1 - p_high) / (1 - p_low))
    log_odds = log_r + math.log(prior_high / (1 - prior_high))
    return 1 / (1 + math.exp(-log_odds))


def posterior_mean_obp(k, n, p_low=0.300, p_high=0.360, prior_high=0.5):
    """Posterior-expected true OBP: a shrunk estimate between the two levels."""
    p = posterior_high(k, n, p_low, p_high, prior_high)
    return p * p_high + (1 - p) * p_low


if __name__ == "__main__":
    print("PA   on-base  P(.360 talent)  est. true OBP")
    for n in (120, 240, 360, 480, 600):
        k = round(0.35 * n)
        print(f"{n:<4} {k:<8} {posterior_high(k, n):>13.1%}  {posterior_mean_obp(k, n):>12.3f}")

"""
Statistical utilities for watermark detection.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom


def f1_score(precision, recall):
    """
    Calculate F1 score given precision and recall arrays.
    
    Args:
        precision: Precision values (array)
        recall: Recall values (array)
    
    Returns:
        np.array: F1 scores
    """
    f1_scores = np.divide(
        2 * precision * recall, 
        precision + recall, 
        where=(precision + recall) != 0
    )
    return f1_scores


def binomial_test_greater(x: int, n: int, p0: float) -> float:
    """
    Exact one-sided binomial test: p-value = P[X >= x | n, p0]
    
    Args:
        x: Number of successes
        n: Total trials
        p0: Null hypothesis probability
        
    Returns:
        float: p-value
    """
    p_val = 0.0
    for k in range(x, n + 1):
        p_val += math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))
    return min(1.0, max(0.0, p_val))


@lru_cache(maxsize=4096)
def bernoulli_mixture_e_value(
    successes: int,
    trials: int,
    p0: float,
    *,
    grid_size: int = 200,
) -> float:
    """Return an anytime-valid e-value for testing Bernoulli success > ``p0``.

    The e-process averages fixed-alternative likelihood-ratio martingales over
    an evenly spaced grid in ``(p0, 1)``. It is valid under optional stopping
    whenever each selected trial has conditional null success probability at
    most ``p0``. Callers should therefore use a shared conservative null bound
    when probe selection is adaptive.
    """

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Expected 0 <= successes <= trials")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be strictly between 0 and 1")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if trials == 0:
        return 1.0

    epsilon = min(1e-6, (1.0 - p0) / (grid_size + 1))
    alternatives = np.linspace(p0 + epsilon, 1.0 - epsilon, grid_size)
    failures = trials - successes
    log_ratios = (
        successes * np.log(alternatives / p0)
        + failures * np.log((1.0 - alternatives) / (1.0 - p0))
    )
    log_e_value = float(logsumexp(log_ratios) - math.log(grid_size))
    return float(math.exp(min(log_e_value, 700.0)))



def calculate_neg_log_p(n, k, p0):
    # 1. Identify the range of the "tail" (from k to n)
    #    In your case: 229, 230, ..., 250
    tail_range = np.arange(k, n + 1)
    
    # 2. Calculate the Log PMF for each value in the tail
    #    This gives us [log(P(229)), log(P(230)), ...]
    #    These values are safe (e.g., -980.5) and won't underflow.
    log_pmfs = binom.logpmf(tail_range, n, p0)
    
    # 3. Sum them safely in log-space
    #    This performs log(exp(p1) + exp(p2) + ...) without leaving log-space
    log_p_value_natural = logsumexp(log_pmfs)
    
    # 4. Convert to base-10 scale for readability
    neg_log10_p = -(log_p_value_natural / np.log(10))
    
    return neg_log10_p

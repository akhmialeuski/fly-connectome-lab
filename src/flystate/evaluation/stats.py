"""Sampling intervals with explicit independent units and seeded resampling."""

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, chi2

from flystate.hashing import stable_int

BOOTSTRAP_CHUNK_SIZE: int = 256


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute a two-sided Wilson score interval for a binomial proportion.

    :param successes: Number of successful independent observations.
    :type successes: int
    :param n: Positive number of observations.
    :type n: int
    :param confidence: Confidence level strictly between zero and one.
    :type confidence: float
    :returns: Lower and upper probability bounds.
    :rtype: tuple[float, float]
    :raises ValueError: If counts or confidence are invalid.
    """
    if n < 1 or not 0 <= successes <= n or not 0 < confidence < 1:
        raise ValueError('Wilson interval requires valid counts and confidence in (0,1).')
    z = NormalDist().inv_cdf(p=(1 + confidence) / 2)
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_mean_interval(
    values: NDArray, samples: int, seed: int, namespace: str = 'design-check'
) -> tuple[float, float]:
    """Bootstrap a mean by resampling independent image-level values.

    :param values: Finite scalar values per independent image, shape (N,), N>0.
    :type values: NDArray
    :param samples: Positive number of bootstrap resamples.
    :type samples: int
    :param seed: Nonnegative explicit resampling seed.
    :type seed: int
    :param namespace: Stable child-stream identifier.
    :type namespace: str
    :returns: 2.5th and 97.5th percentiles using NumPy's linear interpolation.
    :rtype: tuple[float, float]
    :raises ValueError: If observations or resampling settings are invalid.
    """
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or samples < 1
        or seed < 0
    ):
        raise ValueError(
            'Bootstrap requires finite observations, positive samples, and a nonnegative seed.'
        )
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(entropy=[seed, stable_int(key=namespace)])
    )
    means = np.empty(shape=samples, dtype=np.float64)
    for start in range(0, samples, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, samples)
        indices = generator.integers(low=0, high=len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(a=means, q=[2.5, 97.5], method='linear')
    return float(low), float(high)


def mcnemar(n01: int, n10: int) -> dict[str, float | int | str]:
    """Test paired classification discordances with the specified small-count rule.

    :param n01: Images correct in A only.
    :type n01: int
    :param n10: Images correct in B only.
    :type n10: int
    :returns: Discordance counts, two-sided p-value, and method.
    :rtype: dict[str, float | int | str]
    :raises ValueError: If either count is negative or not an integer.
    """
    if type(n01) is not int or type(n10) is not int or min(n01, n10) < 0:
        raise ValueError('McNemar discordances must be nonnegative integers.')
    total = n01 + n10
    method = 'exact' if total < 25 else 'chi2_cc'
    if total == 0:
        probability = 1.0
    elif total < 25:
        probability = float(binomtest(k=n01, n=total, p=0.5).pvalue)
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / total
        probability = float(chi2.sf(x=statistic, df=1))
    return {'n01': n01, 'n10': n10, 'p': probability, 'method': method}


def paired_bootstrap_diff(
    correct_a: NDArray, correct_b: NDArray, samples: int, seed: int
) -> dict[str, float]:
    """Bootstrap paired correctness differences over independent sample identities.

    :param correct_a: Boolean correctness vector for run A, shape (N,).
    :type correct_a: NDArray
    :param correct_b: Boolean correctness vector for run B in the same order.
    :type correct_b: NDArray
    :param samples: Positive number of bootstrap resamples.
    :type samples: int
    :param seed: Explicit nonnegative bootstrap seed.
    :type seed: int
    :returns: Observed difference and percentile interval in percentage points.
    :rtype: dict[str, float]
    :raises ValueError: If paired vector shapes or types are invalid.
    """
    if (
        correct_a.ndim != 1
        or correct_b.shape != correct_a.shape
        or correct_a.dtype != np.bool_
        or correct_b.dtype != np.bool_
    ):
        raise ValueError('Paired correctness must be equally sized boolean vectors.')
    differences = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    low, high = bootstrap_mean_interval(
        values=differences, samples=samples, seed=seed, namespace='compare'
    )
    return {
        'diff_pp': float(differences.mean() * 100),
        'ci_low_pp': low * 100,
        'ci_high_pp': high * 100,
    }

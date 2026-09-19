"""Sampling intervals with explicit independent units and seeded resampling."""

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

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

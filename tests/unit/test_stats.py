"""Check paired scientific statistics against reference distributions and seeded resampling."""

import numpy as np
import pytest
from scipy.stats import binomtest, chi2

from flystate.evaluation.stats import mcnemar, paired_bootstrap_diff
from flystate.hashing import stable_int


class TestPairedStatistics:
    """Exercise exact, asymptotic, and bootstrap branches over independent image pairs."""

    def test_mcnemar_references(self) -> None:
        """Match the zero-discordance case and both prescribed reference distributions."""
        assert mcnemar(n01=0, n10=0) == {'n01': 0, 'n10': 0, 'p': 1.0, 'method': 'exact'}
        exact = mcnemar(n01=3, n10=10)
        assert exact['p'] == binomtest(k=3, n=13, p=0.5).pvalue and exact['method'] == 'exact'
        asymptotic = mcnemar(n01=20, n10=40)
        assert asymptotic['p'] == chi2.sf(x=19**2 / 60, df=1)
        assert asymptotic['method'] == 'chi2_cc'
        assert mcnemar(n01=12, n10=12)['method'] == 'exact'
        assert mcnemar(n01=12, n10=13)['method'] == 'chi2_cc'
        for first, second in ((-1, 0), (0, -1)):
            with pytest.raises(expected_exception=ValueError):
                mcnemar(n01=first, n10=second)

    def test_paired_bootstrap(self) -> None:
        """Verify identical predictions, deterministic seeds, and exact shared-index resampling."""
        a = np.asarray(a=[True, True, False, True, False, True, True, False, False, True])
        b = np.asarray(a=[False, True, True, True, False, False, False, True, False, True])
        zero = paired_bootstrap_diff(correct_a=a, correct_b=a, samples=101, seed=0)
        assert zero == {'diff_pp': 0.0, 'ci_low_pp': 0.0, 'ci_high_pp': 0.0}
        result = paired_bootstrap_diff(correct_a=a, correct_b=b, samples=301, seed=0)
        assert result == paired_bootstrap_diff(correct_a=a, correct_b=b, samples=301, seed=0)
        generator = np.random.Generator(
            bit_generator=np.random.PCG64(
                seed=np.random.SeedSequence(entropy=[0, stable_int(key='compare')])
            )
        )
        indices = generator.integers(low=0, high=len(a), size=(301, len(a)))
        values = (a.astype(float) - b.astype(float))[indices].mean(axis=1)
        low, high = np.percentile(a=values, q=[2.5, 97.5], method='linear')
        assert result == {
            'diff_pp': float((a.astype(float) - b.astype(float)).mean() * 100),
            'ci_low_pp': low * 100,
            'ci_high_pp': high * 100,
        }
        assert paired_bootstrap_diff(
            correct_a=a, correct_b=b, samples=11, seed=1
        ) != paired_bootstrap_diff(correct_a=a, correct_b=b, samples=11, seed=2)
        for first, second in ((a[:, None], b), (a, b[:2]), (a.astype(int), b), (a, b.astype(int))):
            with pytest.raises(expected_exception=ValueError):
                paired_bootstrap_diff(correct_a=first, correct_b=second, samples=100, seed=0)

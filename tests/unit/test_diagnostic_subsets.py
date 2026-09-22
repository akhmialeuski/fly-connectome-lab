"""Training membership isolation and exact fold metadata for learning curves."""

from itertools import pairwise

import numpy as np
import pytest

from flystate.datasets.subset import Sample
from flystate.diagnostics.subsets import subset_protocol, training_subset


class TestTrainingSubset:
    """Check nested draws and prevent held-out photographs from entering fitting."""

    def test_nested_and_order_independent(self) -> None:
        """Construct shuffled cohorts, verify nested membership, and preserve full fitting order."""
        samples = [
            Sample(
                sample_id=f'{label}-{i}',
                filename=f'{label}-{i}.jpg',
                identity=label,
                label=label,
                split='train' if i < 14 else 'val',
            )
            for label in (3, 7)
            for i in range(17)
        ]
        memberships = []
        for count in (2, 4, 8, 14):
            rows = training_subset(samples=samples, per_class=count, seed=0)
            assert len(rows) == count * 2 and np.all(np.diff(a=rows) > 0)
            assert all(samples[i].split == 'train' for i in rows)
            memberships.append({samples[i].sample_id for i in rows})
            reversed_samples = list(reversed(samples))
            reversed_rows = training_subset(samples=reversed_samples, per_class=count, seed=0)
            assert memberships[-1] == {reversed_samples[i].sample_id for i in reversed_rows}
        assert all(left < right for left, right in pairwise(memberships))
        full = training_subset(samples=samples, per_class=14, seed=4)
        np.testing.assert_array_equal(
            actual=full, desired=[i for i, s in enumerate(samples) if s.split == 'train']
        )
        other = training_subset(samples=samples, per_class=2, seed=1)
        assert {samples[i].sample_id for i in other} != memberships[0]
        repeated = training_subset(samples=samples, per_class=2, seed=0)
        assert {samples[i].sample_id for i in repeated} == memberships[0]

    def test_invalid(self) -> None:
        """Reject invalid sizes, seeds, repeated IDs, missing classes, and inadequate support."""
        samples = [
            Sample(sample_id=str(i), filename=str(i), identity=i % 2, label=i % 2, split='train')
            for i in range(4)
        ]
        for records, count, seed in (
            (samples, 1, 0),
            (samples, 2, -1),
            (samples, 3, 0),
            (samples + samples, 2, 0),
            (samples[:1], 2, 0),
            ([], 2, 0),
        ):
            with pytest.raises(expected_exception=ValueError):
                training_subset(samples=records, per_class=count, seed=seed)

    @pytest.mark.parametrize('components', [None, 60])
    def test_fold_metadata(self, components: int | None) -> None:
        """Check bounded fold count and PCA dimensions, with disjoint exhaustive fold IDs.

        :param components: PCA cap or scaling-only mode.
        :type components: Optional[int]
        """
        samples = [
            Sample(sample_id=str(i), filename=str(i), identity=i % 2, label=i % 2, split='train')
            for i in range(8)
        ]
        rows = training_subset(samples=samples, per_class=2, seed=0)
        report = subset_protocol(
            samples=samples, rows=rows, cv_folds=5, seed=0, components=components, feature_count=10
        )
        assert report['effective_cv_folds'] == 2
        assert report['final_pca_components'] == (None if components is None else 3)
        seen = []
        for fold in report['folds']:
            assert fold['training_rows'] == fold['held_out_rows'] == 2
            assert fold['pca_components'] == (None if components is None else 1)
            assert not set(fold['training_sample_ids']) & set(fold['held_out_sample_ids'])
            seen.extend(fold['held_out_sample_ids'])
        assert sorted(seen) == sorted(report['training_sample_ids'])

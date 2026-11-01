"""Verify balanced sampling and the geometry of train-only similarity alignment."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from flystate.datasets.align import (
    align_face,
    canonical_template,
    map_landmarks,
    similarity_transform,
)
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.errors import DatasetError
from flystate.datasets.subset import select_samples
from flystate.experiments.config import PreprocessConfig, SplitConfig, SubsetConfig


class TestSubset:
    """Verify deterministic balanced splits and disjoint calibration identities."""

    def test_balanced_selection(self, synthetic_celeba_dir: Path) -> None:
        """Select exactly 14/3/3 images per class and keep calibration identities disjoint.

        :param synthetic_celeba_dir: Eight-identity offline dataset.
        :type synthetic_celeba_dir: Path
        """
        records = CelebAAdapter(root=synthetic_celeba_dir, expected=None).records()
        subset = SubsetConfig(n_identities=4, calibration_identities=2)
        samples = select_samples(records=records, subset=subset, split=SplitConfig())
        assert samples == select_samples(
            records=list(reversed(records)), subset=subset, split=SplitConfig()
        )
        changed = subset.model_copy(update={'selection_seed': 1})
        assert samples != select_samples(records=records, subset=changed, split=SplitConfig())
        assert len(samples) == len({sample.filename for sample in samples}) == 120
        counts = Counter((sample.label, sample.split) for sample in samples)
        for label in range(4):
            assert [counts[label, split] for split in ('train', 'val', 'test')] == [14, 3, 3]
        assert counts[-1, 'calibration'] == 40
        classification = {sample.identity for sample in samples if sample.label >= 0}
        calibration = {sample.identity for sample in samples if sample.label == -1}
        assert not classification & calibration
        assert {sample.label for sample in samples} == {-1, 0, 1, 2, 3}
        with pytest.raises(expected_exception=DatasetError, match='eligible identities'):
            select_samples(
                records=records, subset=SubsetConfig(n_identities=9), split=SplitConfig()
            )
        with pytest.raises(expected_exception=DatasetError, match='Duplicate sample'):
            select_samples(records=[*records, records[0]], subset=subset, split=SplitConfig())
        with pytest.raises(expected_exception=DatasetError, match='nonempty'):
            select_samples(
                records=records, subset=subset, split=SplitConfig(fractions=(0.98, 0.01, 0.01))
            )


class TestAlignment:
    """Recover known geometry and reject degenerate mappings before image warping."""

    def test_known_transform(self) -> None:
        """Recover a 1.3-scale, 17-degree transform and map all five landmarks accurately."""
        points = np.array(object=[[0, 0], [3, 0], [1, 1], [0, 3], [4, 2]], dtype=np.float64)
        angle = np.deg2rad(17)
        linear = 1.3 * np.array(
            object=[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        target = points @ linear.T + [5, -3]
        actual = similarity_transform(src=points, dst=target)
        np.testing.assert_allclose(
            actual=actual, desired=np.column_stack(tup=(linear, [5, -3])), atol=1e-6
        )
        np.testing.assert_allclose(
            actual=map_landmarks(landmarks=points, transform=actual), desired=target, atol=1e-6
        )
        image = np.full(shape=(32, 32, 3), fill_value=127, dtype=np.uint8)
        identity = align_face(image=image, landmarks=points, template=points, size=32)
        np.testing.assert_array_equal(actual=identity, desired=image)
        reflected = points * [-1, 1]
        fitted = similarity_transform(src=points, dst=reflected)
        assert np.linalg.det(a=fitted[:, :2]) > 0

    def test_template(self) -> None:
        """Place mean eyes at the configured midpoint and interocular distance."""
        points = np.array(
            object=[[69, 109], [106, 113], [87, 132], [73, 152], [108, 154]], dtype=np.float64
        )
        template = canonical_template(
            train_landmarks=np.stack(arrays=[points, points + 5]), pre=PreprocessConfig()
        )
        np.testing.assert_allclose(actual=(template[0] + template[1]) / 2, desired=[64, 52])
        assert np.linalg.norm(x=template[0] - template[1]) == pytest.approx(expected=40)

    def test_invalid_geometry(self) -> None:
        """Reject incorrect dimensions, nonfinite values, coincident eyes and degenerate scale."""
        zeros = np.zeros(shape=(5, 2), dtype=np.float64)
        for invalid in (np.zeros(shape=(4, 2)), np.full(shape=(5, 2), fill_value=np.nan), zeros):
            with pytest.raises(expected_exception=ValueError):
                similarity_transform(src=invalid, dst=zeros)
        for invalid_training in (
            np.zeros(shape=(0, 5, 2)),
            np.zeros(shape=(5, 2)),
            np.zeros(shape=(1, 5, 2)),
        ):
            with pytest.raises(expected_exception=ValueError):
                canonical_template(train_landmarks=invalid_training, pre=PreprocessConfig())
        with pytest.raises(expected_exception=ValueError, match='RGB uint8'):
            align_face(
                image=np.zeros(shape=(3, 3), dtype=np.uint8),
                landmarks=zeros,
                template=zeros,
                size=3,
            )
        with pytest.raises(expected_exception=ValueError, match='affine transform'):
            map_landmarks(landmarks=zeros, transform=np.eye(3))
        symmetric = np.array(object=[[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.float64)
        with pytest.raises(expected_exception=ValueError, match='degenerate scale'):
            similarity_transform(src=symmetric, dst=symmetric * [-1, 1])

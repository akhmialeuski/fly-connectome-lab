"""Nested training-only samples for controlled recognition learning curves."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from flystate.datasets.subset import Sample
from flystate.hashing import sha256_obj, stable_int


def training_subset(samples: Sequence[Sample], per_class: int, seed: int) -> NDArray[np.int64]:
    """Select nested balanced training rows while preserving their original fitting order.

    :param samples: Original cohort records; held-out records never enter the draw.
    :type samples: Sequence[Sample]
    :param per_class: At least two independent training photographs per label.
    :type per_class: int
    :param seed: Nonnegative subset seed, independent of simulation and CV seeds.
    :type seed: int
    :returns: Original row indices (N,) in ascending order.
    :rtype: NDArray[np.int64]
    :raises ValueError: If identifiers repeat, support is insufficient, or settings are invalid.
    """
    if per_class < 2 or seed < 0:
        raise ValueError('Use at least two training photographs per class and a nonnegative seed.')
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError('Sample identifiers must be unique.')
    labels = sorted({sample.label for sample in samples if sample.split == 'train'})
    if len(labels) < 2:
        raise ValueError('A training subset requires at least two classes.')
    selected: list[int] = []
    for label in labels:
        candidates = sorted(
            (
                i
                for i, sample in enumerate(samples)
                if sample.split == 'train' and sample.label == label
            ),
            key=lambda i: samples[i].sample_id,
        )
        if len(candidates) < per_class:
            raise ValueError(f'Class {label} has fewer than {per_class} training photographs.')
        generator = np.random.default_rng(
            seed=np.random.SeedSequence(
                entropy=[seed, stable_int(key=f'learning-curve-subset:label:{label}')]
            )
        )
        selected.extend(generator.permutation(x=candidates)[:per_class].tolist())
    return np.asarray(a=sorted(selected), dtype=np.int64)


def subset_protocol(
    samples: Sequence[Sample],
    rows: NDArray[np.int64],
    cv_folds: int,
    seed: int,
    components: int | None,
    feature_count: int,
) -> dict[str, Any]:
    """Record selected membership and the exact fold sizes and PCA caps before fitting.

    :param samples: Source cohort records.
    :type samples: Sequence[Sample]
    :param rows: Selected training rows in fitting order.
    :type rows: NDArray[np.int64]
    :param cv_folds: Requested fold count from the unchanged readout configuration.
    :type cv_folds: int
    :param seed: Original CV shuffle seed, not the subset seed.
    :type seed: int
    :param components: PCA cap, or null for scaling alone.
    :type components: Optional[int]
    :param feature_count: Number of raw representation features.
    :type feature_count: int
    :returns: Membership hashes, effective fold count, and per-fold sample identifiers.
    :rtype: dict[str, Any]
    """
    ids = [samples[i].sample_id for i in rows]
    labels = np.asarray(a=[samples[i].label for i in rows], dtype=np.int64)
    _, counts = np.unique(ar=labels, return_counts=True)
    folds = StratifiedKFold(
        n_splits=min(cv_folds, int(counts.min())), shuffle=True, random_state=seed
    )
    records = []
    for train, held_out in folds.split(X=rows, y=labels):
        records.append(
            {
                'training_sample_ids': [ids[i] for i in train],
                'held_out_sample_ids': [ids[i] for i in held_out],
                'training_rows': len(train),
                'held_out_rows': len(held_out),
                'pca_components': None
                if components is None
                else min(components, len(train) - 1, feature_count),
            }
        )
    return {
        'training_sample_ids': ids,
        'membership_sha256': sha256_obj(obj=sorted(ids)),
        'ordered_membership_sha256': sha256_obj(obj=ids),
        'effective_cv_folds': folds.n_splits,
        'folds': records,
        'final_pca_components': None
        if components is None
        else min(components, len(rows) - 1, feature_count),
    }

"""Deterministic balanced identity selection and disjoint per-identity splits."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from flystate.datasets.celeba import CelebARecord, sample_id_for
from flystate.datasets.errors import DatasetError
from flystate.experiments.config import SplitConfig, SubsetConfig
from flystate.hashing import stable_int

SplitName = Literal['train', 'val', 'test', 'calibration']
SPLIT_ORDER: tuple[SplitName, ...] = ('train', 'val', 'test', 'calibration')


@dataclass(frozen=True)
class Sample:
    """One selected image with a stable identifier, class label, and split."""

    sample_id: str
    filename: str
    identity: int
    label: int
    split: SplitName


def select_samples(
    records: Sequence[CelebARecord], subset: SubsetConfig, split: SplitConfig
) -> list[Sample]:
    """Select identities and images using the specified single seeded draw sequence.

    :param records: Immutable source annotation records.
    :type records: Sequence[CelebARecord]
    :param subset: Balanced classification and calibration population settings.
    :type subset: SubsetConfig
    :param split: Per-identity train, validation, and test fractions.
    :type split: SplitConfig
    :returns: Samples sorted by split order and sample identifier.
    :rtype: list[Sample]
    :raises DatasetError: If filenames repeat, identities are insufficient, or splits are empty.
    """
    grouped: dict[int, list[str]] = defaultdict(list)
    filenames: set[str] = set()
    for record in records:
        if record.filename in filenames:
            raise DatasetError(f'Duplicate sample filename: {record.filename}.')
        filenames.add(record.filename)
        grouped[record.identity].append(record.filename)
    eligible = sorted(
        identity for identity, names in grouped.items() if len(names) >= subset.images_per_identity
    )
    required = subset.n_identities + subset.calibration_identities
    if len(eligible) < required:
        raise DatasetError(
            f'Need {required} eligible identities; only {len(eligible)} are available.'
        )
    count = subset.images_per_identity
    n_train = round(split.fractions[0] * count)
    n_val = round(split.fractions[1] * count)
    if min(n_train, n_val, count - n_train - n_val) < 1:
        raise DatasetError('Per-identity rounding must leave nonempty train, val, and test splits.')
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(
            entropy=[subset.selection_seed, stable_int(key='celeba-subset')]
        )
    )
    chosen = generator.choice(a=eligible, size=required, replace=False)
    labels = {
        int(identity): label for label, identity in enumerate(sorted(chosen[: subset.n_identities]))
    }
    result: list[Sample] = []
    for identity in sorted(chosen):
        names = sorted(grouped[int(identity)])
        pick = generator.choice(a=len(names), size=count, replace=False)
        selected = sorted(names[index] for index in pick)
        assignments: list[SplitName] = ['calibration'] * count
        label = labels.get(int(identity), -1)
        if label >= 0:
            order = generator.permutation(x=count)
            for index, position in enumerate(order):
                assignments[int(position)] = (
                    'train' if index < n_train else ('val' if index < n_train + n_val else 'test')
                )
        for filename, assignment in zip(selected, assignments, strict=True):
            result.append(
                Sample(
                    sample_id=sample_id_for(filename=filename),
                    filename=filename,
                    identity=int(identity),
                    label=label,
                    split=assignment,
                )
            )
    return sorted(result, key=lambda item: (SPLIT_ORDER.index(item.split), item.sample_id))

"""Find the first selection seeds whose 100 identities avoid every identity used so far.

The search reads identity annotations only; no image, recording or score is touched. Run it with
``FLYSTATE_HOME`` pointing at the data home that registers the CelebA dataset, from the study
directory::

    uv run python analysis/select_confirmation_cohort.py
"""

import json

from flystate.datasets.celeba import CelebAAdapter, CelebARecord
from flystate.datasets.registry import require_validated
from flystate.datasets.subset import select_samples
from flystate.experiments.config import SplitConfig, SubsetConfig
from flystate.settings import get_paths

DATASET: str = 'celeba'
PHOTOGRAPHS_PER_IDENTITY: int = 20
CONFIRMATION_IDENTITIES: int = 100
FRACTIONS: list[float] = [0.7, 0.15, 0.15]
# (identities, selection seed) of every earlier cohort: T16 main, T20 to T36 development, T35
# confirmation, and the ten T37 confirmation cohorts.
USED_COHORTS: tuple[tuple[int, int], ...] = (
    (100, 0),
    (20, 0),
    (20, 2),
    *((20, seed) for seed in (4, 5, 6, 8, 10, 11, 15, 22, 24, 25)),
)
WANTED: int = 3
LAST_SEED: int = 20000


def identities(records: list[CelebARecord], count: int, seed: int) -> set[int]:
    """Return the identities that the project's subset rule selects.

    :param records: CelebA annotation records.
    :type records: list[CelebARecord]
    :param count: Number of identities.
    :type count: int
    :param seed: Selection seed.
    :type seed: int
    :returns: Selected CelebA identity numbers.
    :rtype: set[int]
    """
    subset = SubsetConfig(
        n_identities=count,
        images_per_identity=PHOTOGRAPHS_PER_IDENTITY,
        calibration_identities=0,
        selection_seed=seed,
    )
    split = SplitConfig(kind='per_identity', fractions=FRACTIONS)
    return {
        sample.identity for sample in select_samples(records=records, subset=subset, split=split)
    }


def main() -> None:
    """Print the used-identity count and the first clean selection seeds as one JSON object."""
    root = require_validated(paths=get_paths(), name=DATASET)
    records = CelebAAdapter(root=root, expected=None).records()
    used: set[int] = set()
    for count, seed in USED_COHORTS:
        used |= identities(records=records, count=count, seed=seed)
    clean = []
    for seed in range(1, LAST_SEED):
        if not identities(records=records, count=CONFIRMATION_IDENTITIES, seed=seed) & used:
            clean.append(seed)
            if len(clean) == WANTED:
                break
    print(json.dumps({'used': len(used), 'clean_seeds': clean}))


if __name__ == '__main__':
    main()

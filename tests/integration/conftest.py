"""Small complete offline experiment shared by trace and readout integration tests."""

import shutil
from pathlib import Path

import pytest

from flystate.datasets import registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.experiments.config import ExperimentConfig
from flystate.settings import get_paths
from flystate.traces import store


@pytest.fixture
def tiny_experiment(
    synthetic_celeba_dir: Path, synthetic_brain_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> ExperimentConfig:
    """Install synthetic data and a small brain with three trace chunks.

    :param synthetic_celeba_dir: Synthetic CelebA source.
    :type synthetic_celeba_dir: Path
    :param synthetic_brain_dir: Synthetic connectome source.
    :type synthetic_brain_dir: Path
    :param monkeypatch: Scoped chunk-size override.
    :type monkeypatch: pytest.MonkeyPatch
    :returns: Four-identity experiment requiring no network.
    :rtype: ExperimentConfig
    """
    paths = get_paths()
    shutil.copytree(src=synthetic_brain_dir, dst=paths.brain)
    registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
    registry.set_validation(
        paths=paths,
        name='celeba',
        full=True,
        report=CelebAAdapter(root=synthetic_celeba_dir, expected=None).validate(full=True),
    )
    monkeypatch.setattr(target=store, name='CHUNK_SIZE', value=16)
    return ExperimentConfig.model_validate(
        obj={
            'name': 'tiny',
            'dataset': {'subset': {'n_identities': 4, 'images_per_identity': 10}},
            'episodes': {'steps': 4, 'trajectory': {'strategy': 'raster'}},
            'encoder': {'neurons_per_channel': 64, 'position': {'neurons_per_axis': 4}},
            'brain': {'threads': 1, 'batch_size': 3, 'warmup_steps': 4, 'steps_per_observation': 2},
            'readout': {'pca_components': 10, 'c_grid': [0.01, 0.1], 'cv_folds': 2},
            'evaluation': {'bootstrap_samples': 100},
        }
    )

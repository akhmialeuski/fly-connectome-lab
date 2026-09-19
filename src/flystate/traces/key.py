"""Simulation identity includes resolved threads and excludes execution batching."""

from typing import Any, Literal

import flybrain

from flystate.datasets.preprocess import PreparedDataset
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_obj

TRACE_FORMAT: int = 1


def simulation_mode(cfg: ExperimentConfig) -> Literal['persistent', 'reset']:
    """Map readout concatenation to its reusable reset simulation.

    :param cfg: Validated experiment configuration.
    :type cfg: ExperimentConfig
    :returns: Actual simulation state policy.
    :rtype: Literal['persistent', 'reset']
    """
    return 'persistent' if cfg.memory.mode == 'persistent' else 'reset'


def cache_key_fields(
    cfg: ExperimentConfig, prepared: PreparedDataset, brain_files_sha: dict[str, str], threads: int
) -> dict[str, Any]:
    """Describe all inputs that can change a cached neural feature value.

    :param cfg: Validated experiment settings.
    :type cfg: ExperimentConfig
    :param prepared: Content-verified aligned dataset.
    :type prepared: PreparedDataset
    :param brain_files_sha: Exact loaded connectome artifact digests.
    :type brain_files_sha: dict[str, str]
    :param threads: Resolved Numba accumulation thread count.
    :type threads: int
    :returns: Canonical-JSON-compatible simulation identity.
    :rtype: dict[str, Any]
    :raises ValueError: If the resolved thread count is invalid.
    """
    if threads < 1:
        raise ValueError('Cache identity requires a positive resolved thread count.')
    brain = cfg.brain.model_dump(mode='json', exclude={'batch_size', 'threads'})
    brain['threads'] = threads
    return {
        'format': TRACE_FORMAT,
        'dataset_fingerprint': prepared.fingerprint,
        'dataset': cfg.dataset.model_dump(mode='json'),
        'episodes': cfg.episodes.model_dump(mode='json'),
        'encoder': cfg.encoder.model_dump(mode='json'),
        'brain': brain,
        'simulation_mode': simulation_mode(cfg=cfg),
        'readout': cfg.readout.model_dump(
            mode='json', include={'population', 'features', 'trace_tau_s'}
        ),
        'seed': cfg.seed,
        'brain_files_sha256': brain_files_sha,
        'flybrain_version': flybrain.__version__,
    }


def cache_key(fields: dict[str, Any]) -> str:
    """Hash a fully resolved simulation description.

    :param fields: JSON-compatible identity fields.
    :type fields: dict[str, Any]
    :returns: SHA-256 cache directory basename.
    :rtype: str
    """
    return sha256_obj(obj=fields)

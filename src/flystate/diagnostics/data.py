"""Matched development-only representations from verified images and neural traces."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flystate.datasets.preprocess import prepare_dataset
from flystate.datasets.subset import Sample
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.readouts.training import read_feature_rows
from flystate.settings import Paths
from flystate.traces.builder import open_trace

REPRESENTATIONS: tuple[str, ...] = ('pixels', 'encoded', 'neural')
HISTORIES: tuple[str, ...] = ('last', 'all')
FEATURE_BLOCKS: tuple[str, ...] = ('both', 'spike_trace', 'voltage')


@dataclass(frozen=True)
class ProbeData:
    """Float32 development features (N,F), ordered sample records, and source provenance."""

    x: NDArray[np.float32]
    samples: list[Sample]
    provenance: dict[str, Any]


def load_representation(
    cfg: ExperimentConfig,
    paths: Paths,
    representation: str,
    history: str,
    features: str,
) -> ProbeData:
    """Load only train/validation rows with matched observation and neuron identities.

    :param cfg: Original experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home containing verified caches.
    :type paths: Paths
    :param representation: Pixels, encoded input, or neural features.
    :type representation: str
    :param history: Last observation or concatenated available history.
    :type history: str
    :param features: Both neural blocks or one named block.
    :type features: str
    :returns: Development feature matrix, records, and immutable source identities.
    :rtype: ProbeData
    :raises ValueError: If representation or geometry is unsupported.
    """
    if (
        representation not in REPRESENTATIONS
        or history not in HISTORIES
        or features not in FEATURE_BLOCKS
        or (representation != 'neural' and features != 'both')
    ):
        raise ValueError('Unsupported diagnostic representation, history, or feature selection.')
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    rows = np.asarray(
        a=[i for i, sample in enumerate(prepared.samples) if sample.split in {'train', 'val'}],
        dtype=np.int64,
    )
    samples = [prepared.samples[i] for i in rows]
    provenance: dict[str, Any] = {
        'dataset_fingerprint': prepared.fingerprint,
        'preprocessing_key': prepared.key,
        'config_hash': config_hash(cfg=cfg),
        'test_rows_loaded_for_scoring': False,
    }
    if representation == 'neural':
        store = open_trace(cfg=cfg, paths=paths)
        values = read_feature_rows(store=store, rows=rows)
        if features != 'both':
            if features not in cfg.readout.features:
                raise ValueError('Requested feature block was not recorded in this trace.')
            values = np.split(ary=values, indices_or_sections=len(cfg.readout.features), axis=2)[
                cfg.readout.features.index(features)
            ]
        provenance['cache_key'] = store.meta['key']
        provenance['brain_files_sha256'] = store.meta['key_fields']['brain_files_sha256']
    else:
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as brain:
            candidates = np.flatnonzero(
                a=(brain['superclass'] == cfg.encoder.target_population)
                | (brain['cell_type'] == cfg.encoder.target_population)
            )
        encoder = SparseProjectionEncoder(
            cfg=cfg.encoder, window=cfg.episodes.window, candidate_neurons=candidates
        )
        provenance['encoder'] = encoder.spec()
        provenance['brain_metadata_sha256'] = sha256_file(path=paths.brain / 'brain.npz')
        feature_count = (
            cfg.episodes.window**2 * 3 if representation == 'pixels' else len(encoder.input_idx)
        )
        steps = cfg.episodes.steps if history == 'all' else 1
        values = np.empty(shape=(len(samples), steps, feature_count), dtype=np.float32)
        for offset, row in enumerate(rows):
            episode = builder.build(sample=samples[offset], image=prepared.images[row])
            observations = episode.observations if history == 'all' else episode.observations[-1:]
            positions = episode.positions if history == 'all' else episode.positions[-1:]
            values[offset] = (
                observations.reshape(steps, -1).astype(np.float32) / 255
                if representation == 'pixels'
                else encoder.encode(observations=observations, positions=positions).T
            )
    x = values[:, -1, :] if history == 'last' else values.reshape(len(samples), -1)
    return ProbeData(x=x, samples=samples, provenance=provenance)

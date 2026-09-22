"""Training-only paired noise interventions and native storage-precision measurements."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.runtime import EpisodeBrain, RestState
from flystate.diagnostics.artifacts import attempt
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import Episode, EpisodeBuilder
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table
from flystate.traces.builder import open_trace, resolve_trace

COMMON_NOISE_ID: str = 'signal-stability-common-noise'


def record_responses(
    brain: EpisodeBrain,
    rest: RestState,
    encoder: SparseProjectionEncoder,
    episodes: list[Episode],
    cfg: ExperimentConfig,
    noise_seed: int,
    common_noise: bool,
    blank: bool,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Record native features and spike counts with independent image and noise identifiers.

    :param brain: Fixed-thread simulation runtime.
    :type brain: EpisodeBrain
    :param rest: Shared baseline warmed state, unchanged across interventions.
    :type rest: RestState
    :param encoder: Original fixed image-to-neuron projection.
    :type encoder: SparseProjectionEncoder
    :param episodes: Ordered training image episodes; one reference episode for blank controls.
    :type episodes: list[Episode]
    :param cfg: Original timing and feature definitions.
    :type cfg: ExperimentConfig
    :param noise_seed: Episode-noise seed only; does not change encoder or rest.
    :type noise_seed: int
    :param common_noise: Use the same stream identifier for every image.
    :type common_noise: bool
    :param blank: Inject exactly zero current, including positional channels.
    :type blank: bool
    :returns: Float32 features (N,T,F) and int64 spike summaries (N,T,3).
    :rtype: tuple[NDArray[np.float32], NDArray[np.int64]]
    :raises ValueError: If generated features are nonfinite.
    """
    values = np.empty(
        shape=(
            len(episodes),
            cfg.episodes.steps,
            len(brain.readout_idx) * len(cfg.readout.features),
        ),
        dtype=np.float32,
    )
    counts = np.empty(shape=(*values.shape[:2], 3), dtype=np.int64)
    for start in range(0, len(episodes), brain.batch_size):
        batch = episodes[start : start + brain.batch_size]
        brain.begin(
            rest=rest,
            sample_ids=[COMMON_NOISE_ID if common_noise else e.sample_id for e in batch],
            seed=noise_seed,
        )
        for step in range(cfg.episodes.steps):
            currents = np.zeros(shape=(len(encoder.input_idx), brain.batch_size), dtype=np.float32)
            if not blank:
                currents[:, : len(batch)] = encoder.encode(
                    observations=np.stack(arrays=[e.observations[step] for e in batch]),
                    positions=np.stack(arrays=[e.positions[step] for e in batch]),
                )
            summary = brain.run(
                input_idx=encoder.input_idx,
                currents=currents,
                n_steps=cfg.brain.steps_per_observation,
            )
            values[start : start + len(batch), step] = brain.features(kinds=cfg.readout.features)[
                : len(batch)
            ]
            counts[start : start + len(batch), step] = np.stack(
                arrays=[summary.spikes_total, summary.active_neurons, summary.readout_spikes],
                axis=1,
            )[: len(batch)]
    if not np.isfinite(values).all():
        raise ValueError('Native neural responses must be finite.')
    return values, counts


def response_distances(first: NDArray, second: NDArray, labels: NDArray) -> dict[str, Any]:
    """Describe paired and within/between-identity squared distances by observation.

    :param first: Float features (N,T,F) under the first common-noise stream.
    :type first: NDArray
    :param second: Matched ordered responses (N,T,F) under the second stream.
    :type second: NDArray
    :param labels: Identity class labels (N,).
    :type labels: NDArray
    :returns: Descriptive means and quantiles; dependent pairs are not inference replicates.
    :rtype: dict[str, Any]
    :raises ValueError: If geometry or finite-value requirements are violated.
    """
    if (
        first.ndim != 3
        or first.shape != second.shape
        or min(first.shape) < 1
        or labels.shape != (len(first),)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError('Distances require matched finite (N,T,F) responses and N labels.')
    a, b = first.astype(np.float64), second.astype(np.float64)
    groups: dict[str, list[NDArray]] = {
        'same_image_different_noise': list(np.mean(a=(a - b) ** 2, axis=2)),
        'same_identity_different_image': [],
        'different_identity_same_noise': [],
    }
    for i in range(len(a)):
        for j in range(i):
            key = (
                'same_identity_different_image'
                if labels[i] == labels[j]
                else 'different_identity_same_noise'
            )
            groups[key].append(np.mean(a=(a[i] - a[j]) ** 2, axis=1))
    return {
        key: {
            'pairs': len(rows),
            'mean_squared_distance_by_observation': np.mean(a=rows, axis=0).tolist()
            if rows
            else None,
            'quantiles_10_50_90_by_observation': np.quantile(
                a=rows, q=[0.1, 0.5, 0.9], axis=0
            ).T.tolist()
            if rows
            else None,
        }
        for key, rows in groups.items()
    }


def run_stability(
    cfg: ExperimentConfig, paths: Paths, output: Path, sample_ids: list[str], membership_sha256: str
) -> dict[str, Any]:
    """Execute paired training-only controls after exact archived float16 replay succeeds.

    :param cfg: Original persistent-state configuration and native noise baseline.
    :type cfg: ExperimentConfig
    :param paths: Existing data home, connectome, images and completed trace cache.
    :type paths: Paths
    :param output: Fresh immutable diagnostic directory.
    :type output: Path
    :param sample_ids: Frozen ordered unique training photograph identifiers.
    :type sample_ids: list[str]
    :param membership_sha256: Canonical JSON digest of the ordered identifiers.
    :type membership_sha256: str
    :returns: Quantization and descriptive response-distance measurements, without recognition fits.
    :rtype: dict[str, Any]
    :raises ValueError: If membership, protocol, replay or numerical checks fail.
    """
    parameters = {
        'kind': 'signal_stability',
        'hypothesis': 'Measure image-specific signal versus noise and float16 storage loss.',
        'noise_seeds': [0, 1],
        'common_noise_identifier': COMMON_NOISE_ID,
        'membership_sha256': membership_sha256,
        'rest_seed': cfg.seed,
        'trainable_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if (
            not sample_ids
            or len(set(sample_ids)) != len(sample_ids)
            or sha256_obj(obj=sample_ids) != membership_sha256
            or cfg.memory.mode != 'persistent'
            or not cfg.brain.noise.enabled
        ):
            raise ValueError(
                'Require unique frozen training IDs and a noise-enabled persistent baseline.'
            )
        prepared, fields, _ = resolve_trace(cfg=cfg, paths=paths)
        store = open_trace(cfg=cfg, paths=paths)
        by_id = {s.sample_id: i for i, s in enumerate(prepared.samples) if s.split == 'train'}
        if any(s not in by_id for s in sample_ids):
            raise ValueError('Every selected sample must belong to the original training split.')
        rows = [by_id[s] for s in sample_ids]
        samples = [prepared.samples[i] for i in rows]
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        episodes = [
            builder.build(sample=prepared.samples[i], image=prepared.images[i]) for i in rows
        ]
        write_table(path=directory / 'samples.parquet', rows=[asdict(s) for s in samples])
        write_json(
            path=directory / 'provenance.json',
            value={
                'dataset_fingerprint': prepared.fingerprint,
                'cache_key_fields': fields,
                'sample_ids': sample_ids,
                'validation_test_reserve_scored': False,
            },
        )
        original_threads = numba.get_num_threads()
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                brain = EpisodeBrain(
                    brain_dir=paths.brain,
                    brain_cfg=cfg.brain,
                    readout_cfg=cfg.readout,
                    batch_size=cfg.brain.batch_size,
                    threads=fields['brain']['threads'],
                )
                rest = brain.compute_rest_state(seed=cfg.seed)
                encoder = SparseProjectionEncoder(
                    cfg=cfg.encoder,
                    window=cfg.episodes.window,
                    candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
                )
                np.savez(
                    file=directory / 'rest.npz',
                    voltage=rest.v,
                    fired=rest.fired,
                    readout_idx=brain.readout_idx,
                    input_idx=encoder.input_idx,
                )
                responses: dict[str, NDArray] = {}
                cases = [
                    ('native', cfg.seed, False, False),
                    ('matched-0', 0, True, False),
                    ('matched-1', 1, True, False),
                    ('noise-off', 0, True, False),
                    ('blank-0', 0, True, True),
                    ('blank-1', 1, True, True),
                    ('blank-off', 0, True, True),
                ]
                report: dict[str, Any] = {'parameters': parameters, 'cases': [], 'quantization': {}}
                for name, seed, common, blank in cases:
                    disabled = name.endswith('off')
                    case_cfg = cfg.brain.model_copy(
                        update={
                            'noise': cfg.brain.noise.model_copy(update={'enabled': not disabled})
                        }
                    )
                    brain = EpisodeBrain(
                        brain_dir=paths.brain,
                        brain_cfg=case_cfg,
                        readout_cfg=cfg.readout,
                        batch_size=cfg.brain.batch_size,
                        threads=fields['brain']['threads'],
                    )
                    values, counts = record_responses(
                        brain=brain,
                        rest=rest,
                        encoder=encoder,
                        episodes=episodes[:1] if blank else episodes,
                        cfg=cfg,
                        noise_seed=seed,
                        common_noise=common,
                        blank=blank,
                    )
                    filename = directory / f'{name}.npz'
                    np.savez(file=filename, features=values, spike_summaries=counts)
                    responses[name] = values
                    report['cases'].append(
                        {
                            'name': name,
                            'episodes': len(values),
                            'noise_seed': seed,
                            'common_noise': common,
                            'zero_current': blank,
                            'episode_noise_enabled': not disabled,
                            'rest_seed': rest.seed,
                            'sha256': sha256_file(path=filename),
                        }
                    )
                    if name == 'native':
                        archived = np.stack(
                            arrays=[np.asarray(store.array(name='features')[r]) for r in rows]
                        )
                        rounded = values.astype(np.float16)
                        exact = bool(np.array_equal(rounded, archived))
                        report['native_replay_exact'] = exact
                        write_json(path=directory / 'report.json', value=report)
                        if not exact or not np.isfinite(rounded).all():
                            raise ValueError(
                                'Native float16 replay differs from archived rows; '
                                'stop interpretation.'
                            )
                        error = values.astype(np.float64) - rounded.astype(np.float64)
                        report['quantization'] = {
                            'max_absolute_error': float(np.max(a=np.abs(error))),
                            'rms_error_by_observation': np.sqrt(
                                np.mean(a=error**2, axis=(0, 2))
                            ).tolist(),
                        }
                    write_json(path=directory / 'report.json', value=report)
                labels = np.asarray(a=[s.label for s in samples])
                report['distances'] = {
                    'both': response_distances(
                        first=responses['matched-0'], second=responses['matched-1'], labels=labels
                    )
                }
                report['evoked_mean_squared_response'] = {}
                width = len(brain.readout_idx)
                for index, kind in enumerate(cfg.readout.features):
                    block = slice(index * width, (index + 1) * width)
                    quant_error = responses['native'][:, :, block].astype(np.float64) - responses[
                        'native'
                    ][:, :, block].astype(np.float16).astype(np.float64)
                    report['quantization'][kind] = {
                        'max_absolute_error': float(np.max(a=np.abs(quant_error))),
                        'rms_error_by_observation': np.sqrt(
                            np.mean(a=quant_error**2, axis=(0, 2))
                        ).tolist(),
                    }
                    report['evoked_mean_squared_response'][kind] = {
                        name: np.mean(
                            a=(
                                responses[name][:, :, block].astype(np.float64)
                                - responses[blank][:, :, block]
                            )
                            ** 2,
                            axis=(0, 2),
                        ).tolist()
                        for name, blank in [
                            ('matched-0', 'blank-0'),
                            ('matched-1', 'blank-1'),
                            ('noise-off', 'blank-off'),
                        ]
                    }
                    report['distances'][kind] = response_distances(
                        first=responses['matched-0'][:, :, block],
                        second=responses['matched-1'][:, :, block],
                        labels=labels,
                    )
                report['conclusion'] = (
                    'Native replay passed. Paired noise and image responses are recorded; '
                    'these descriptive measurements do not establish recognition '
                    'or a memory benefit.'
                )
                write_json(path=directory / 'report.json', value=report)
                return report
        finally:
            numba.set_num_threads(n=original_threads)

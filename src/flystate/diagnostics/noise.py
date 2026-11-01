"""Verified native development traces with episode noise isolated from warmup and fitting."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numba
import numpy as np
from threadpoolctl import threadpool_limits

from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.subset import Sample
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.data import FEATURE_BLOCKS, HISTORIES, ProbeData
from flystate.diagnostics.stability import record_responses
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table, write_table
from flystate.traces.builder import open_trace, resolve_trace


def generate_noise_trace(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    membership: dict[str, list[str]],
    membership_sha256: str,
    episode_seed: int,
    noise_enabled: bool,
) -> dict[str, Any]:
    """Generate native train/validation responses while holding original warmup fixed.

    :param cfg: Original persistent noise-enabled configuration.
    :type cfg: ExperimentConfig
    :param paths: Existing data home and original verified trace cache.
    :type paths: Paths
    :param output: Fresh diagnostic attempt directory inside the data home.
    :type output: Path
    :param membership: Frozen sorted sample IDs for train and val splits.
    :type membership: dict[str, list[str]]
    :param membership_sha256: Canonical JSON hash of the frozen split membership.
    :type membership_sha256: str
    :param episode_seed: Episode-noise seed only; warmup and classifier seeds stay unchanged.
    :type episode_seed: int
    :param noise_enabled: Enable original episode noise or disable it after baseline warmup.
    :type noise_enabled: bool
    :returns: Trace metadata and replay evidence; no classifier is fitted.
    :rtype: dict[str, Any]
    :raises ValueError: If configuration, membership, finite values or native replay fails.
    """
    parameters = {
        'kind': 'noise_trace',
        'hypothesis': 'Isolate episode noise with fixed warmup.',
        'episode_seed': episode_seed,
        'episode_noise_enabled': noise_enabled,
        'rest_seed': cfg.seed,
        'membership_sha256': membership_sha256,
        'trainable_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if episode_seed < 0 or cfg.memory.mode != 'persistent' or not cfg.brain.noise.enabled:
            raise ValueError(
                'Require a persistent noise-enabled baseline and nonnegative episode seed.'
            )
        prepared, fields, _ = resolve_trace(cfg=cfg, paths=paths)
        store = open_trace(cfg=cfg, paths=paths)
        rows = [i for i, s in enumerate(prepared.samples) if s.split in {'train', 'val'}]
        samples = [prepared.samples[i] for i in rows]
        actual = {
            split: sorted(s.sample_id for s in samples if s.split == split)
            for split in ('train', 'val')
        }
        if actual != membership or sha256_obj(obj=actual) != membership_sha256:
            raise ValueError('Development membership differs from the frozen protocol.')
        write_table(path=directory / 'samples.parquet', rows=[asdict(obj=s) for s in samples])
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        episodes = [
            builder.build(sample=prepared.samples[i], image=prepared.images[i]) for i in rows
        ]
        episode_cfg = cfg.brain.model_copy(
            update={'noise': cfg.brain.noise.model_copy(update={'enabled': noise_enabled})}
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
                if not noise_enabled:
                    brain = EpisodeBrain(
                        brain_dir=paths.brain,
                        brain_cfg=episode_cfg,
                        readout_cfg=cfg.readout,
                        batch_size=cfg.brain.batch_size,
                        threads=fields['brain']['threads'],
                    )
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
                values, counts = record_responses(
                    brain=brain,
                    rest=rest,
                    encoder=encoder,
                    episodes=episodes,
                    cfg=cfg,
                    noise_seed=episode_seed,
                    common_noise=False,
                    blank=False,
                )
                np.savez(file=directory / 'features.npz', features=values, spike_summaries=counts)
                baseline = noise_enabled and episode_seed == cfg.seed
                exact: bool | None = None
                if baseline:
                    archived = np.stack(
                        arrays=[np.asarray(a=store.array(name='features')[r]) for r in rows]
                    )
                    exact = bool(np.array_equal(a1=values.astype(np.float16), a2=archived))
                provenance = {
                    'source_config_sha256': config_hash(cfg=cfg),
                    'dataset_fingerprint': prepared.fingerprint,
                    'cache_key_fields': fields,
                    'episode_brain_config': episode_cfg.model_dump(mode='json'),
                    'parameters': parameters,
                    'membership_sha256': membership_sha256,
                    'ordered_sample_ids_sha256': sha256_obj(obj=[s.sample_id for s in samples]),
                    'features_sha256': sha256_file(path=directory / 'features.npz'),
                    'feature_blocks': cfg.readout.features,
                    'readout_neurons': len(brain.readout_idx),
                    'shape': list(values.shape),
                    'native_replay_exact': exact,
                    'historical_test_and_reserve_simulated': False,
                }
                write_json(path=directory / 'provenance.json', value=provenance)
                report = {
                    'parameters': parameters,
                    'provenance': provenance,
                    'conclusion': f'Recorded {len(samples)} sequential development episodes. '
                    'No recognition score is implied until a separate readout is fitted.',
                }
                write_json(path=directory / 'report.json', value=report)
                if baseline and not exact:
                    raise ValueError(
                        'Native float16 replay differs from the original development cache.'
                    )
                return report
        finally:
            numba.set_num_threads(n=original_threads)


def load_noise_trace(
    cfg: ExperimentConfig, paths: Paths, source: Path, precision: str, history: str, features: str
) -> ProbeData:
    """Load a completed verified native trace with an explicit storage-precision treatment.

    :param cfg: Original configuration, including unchanged classifier seed and membership.
    :type cfg: ExperimentConfig
    :param paths: Data home containing the source trace attempt.
    :type paths: Paths
    :param source: Completed native trace directory within the data home.
    :type source: Path
    :param precision: Native float32 or float16 quantized from that same native trace.
    :type precision: str
    :param history: Last observation or all sequential observations.
    :type history: str
    :param features: Both original blocks or one named block.
    :type features: str
    :returns: Float32 features (N,F), ordered development samples and source provenance.
    :rtype: ProbeData
    :raises ValueError: If source integrity, configuration, membership or geometry is invalid.
    """
    if (
        precision not in {'float32', 'float16'}
        or history not in HISTORIES
        or features not in FEATURE_BLOCKS
    ):
        raise ValueError('Unsupported trace precision, history or feature block.')
    directory = output_path(path=source, paths=paths)
    manifest = verify_attempt_inventory(directory=directory, paths=paths)
    provenance = json.loads(s=(directory / 'provenance.json').read_text(encoding='utf-8'))
    if (
        manifest.get('status') != 'completed'
        or manifest.get('parameters', {}).get('kind') != 'noise_trace'
        or provenance['source_config_sha256'] != config_hash(cfg=cfg)
        or provenance['features_sha256'] != sha256_file(path=directory / 'features.npz')
    ):
        raise ValueError('Trace must be completed and match the original configuration and arrays.')
    samples = [Sample(**row) for row in read_table(path=directory / 'samples.parquet')]
    membership = {
        split: sorted(s.sample_id for s in samples if s.split == split)
        for split in ('train', 'val')
    }
    if (
        any(s.split not in {'train', 'val'} for s in samples)
        or len({s.sample_id for s in samples}) != len(samples)
        or sha256_obj(obj=membership) != provenance['membership_sha256']
        or sha256_obj(obj=[s.sample_id for s in samples]) != provenance['ordered_sample_ids_sha256']
    ):
        raise ValueError('Native trace development membership is inconsistent.')
    with np.load(file=directory / 'features.npz', allow_pickle=False) as arrays:
        values = arrays['features']
    blocks = provenance['feature_blocks']
    width = provenance['readout_neurons']
    if (
        values.dtype != np.float32
        or values.shape != (len(samples), cfg.episodes.steps, width * len(blocks))
        or tuple(blocks) != tuple(cfg.readout.features)
        or width < 1
        or (features != 'both' and features not in blocks)
        or not np.isfinite(values).all()
    ):
        raise ValueError('Native trace feature geometry or finite float32 values are invalid.')
    if features != 'both':
        index = blocks.index(features)
        values = values[:, :, index * width : (index + 1) * width]
    if precision == 'float16':
        values = values.astype(np.float16).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Float16 treatment overflowed; no valid probe can be fitted.')
    x = values[:, -1, :] if history == 'last' else values.reshape(len(samples), -1)
    return ProbeData(
        x=x,
        samples=samples,
        provenance={
            **provenance,
            'source_trace': str(directory.relative_to(paths.home)),
            'source_manifest_sha256': sha256_file(path=directory / 'manifest.json'),
            'trace_precision': precision,
            'selected_history': history,
            'selected_features': features,
        },
    )

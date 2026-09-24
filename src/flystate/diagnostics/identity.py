"""Frozen training-only identity responses in selected fly populations."""

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
import psutil
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.runtime import EpisodeBrain, TemporalResponse
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.temporal import (
    _read_json,
    _verify_cohort,
    _verify_masks,
    _write_arrays,
)
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json

CASES: dict[str, dict[str, Any]] = {
    'N0': {'id': 'N0', 'noise_enabled': True, 'episode_seed': 0},
    'N1': {'id': 'N1', 'noise_enabled': True, 'episode_seed': 1},
    'OFF': {'id': 'OFF', 'noise_enabled': False, 'episode_seed': 0},
}


def _noise_identifier(window: int, case_id: str) -> str:
    """Return the frozen common-noise identifier for an image window.

    :param window: Zero-based raster observation index.
    :type window: int
    :param case_id: Frozen condition identifier.
    :type case_id: str
    :returns: Image-independent noise-stream identifier.
    :rtype: str
    """
    return f'identity-access|window={window}|stream={case_id}'


def _save_response(
    recorded: dict[str, list[NDArray[Any]]], response: TemporalResponse, populations: list[str]
) -> None:
    """Append one episode's numeric states without storing image pixels.

    :param recorded: Mutable attempt-local array buffers.
    :type recorded: dict[str, list[NDArray[Any]]]
    :param response: Checkpoint states for one fly, with arrays (C,1,K).
    :type response: TemporalResponse
    :param populations: Fixed population names in protocol order.
    :type populations: list[str]
    """
    for name in populations:
        recorded.setdefault(f'voltage_{name}', []).append(response.voltages[name][:, 0])
        recorded.setdefault(f'spikes_{name}', []).append(response.spike_counts[name][:, 0])
    for name, values in (
        ('readout_trace', response.readout_traces[:, 0]),
        ('noise_kicks', response.noise_kicks[:, 0]),
        ('noise_digests', response.noise_digests[:, 0]),
        ('total_spikes', response.total_spikes[:, 0]),
        ('active_neurons', response.active_neurons[:, 0]),
    ):
        recorded.setdefault(name, []).append(values)


def _check_schedule(
    cfg: ExperimentConfig,
    schedule: dict[str, Any],
    cohort_path: Path,
    masks_path: Path,
    membership_path: Path,
) -> None:
    """Reject edited or incompatible machine-readable protocol settings.

    :param cfg: Effective source experiment configuration.
    :type cfg: ExperimentConfig
    :param schedule: Frozen case, timing, and resource schedule.
    :type schedule: dict[str, Any]
    :param cohort_path: Selected-image protocol file.
    :type cohort_path: Path
    :param masks_path: Selected-neuron protocol file.
    :type masks_path: Path
    :param membership_path: Original split membership file.
    :type membership_path: Path
    :raises ValueError: If a source digest or frozen experiment parameter differs.
    """
    if (
        schedule.get('schema_version') != 1
        or schedule.get('cases') != list(CASES.values())
        or schedule.get('windows') != [0, 7, 15]
        or schedule.get('checkpoints') != [0, 1, 2, 5, 10, 11, 12, 15, 20]
        or schedule.get('stimulus_steps') != 10
        or schedule.get('recovery_steps') != 10
        or schedule.get('primary_endpoint_step') != 10
        or schedule.get('null_permutations') != 10_000
        or schedule.get('null_seed_key') != 't28-primary-null'
        or schedule.get('common_noise_identifier_template')
        != 'identity-access|window=<window>|stream=<case>'
        or schedule.get('memory_limit_bytes') != 12 * 1024**3
        or schedule.get('campaign_time_limit_seconds') != 3600
        or schedule.get('cohort_sha256') != sha256_file(path=cohort_path)
        or schedule.get('masks_sha256') != sha256_file(path=masks_path)
        or schedule.get('membership_file_sha256') != sha256_file(path=membership_path)
        or cfg.encoder.amplitude != 0.05
        or cfg.brain.dt_s != 0.02
        or cfg.brain.steps_per_observation != 10
        or cfg.brain.warmup_steps != 25
        or cfg.brain.threads != 4
        or cfg.brain.batch_size != 1
        or not cfg.brain.noise.enabled
        or cfg.readout.population != 'descending_neuron'
        or cfg.seed != 0
    ):
        raise ValueError('Identity-access inputs differ from the frozen schedule.')


def run_identity_case(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    case_id: str,
    cohort_path: Path,
    masks_path: Path,
    membership_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    """Record one immutable original-current identity-access condition.

    :param cfg: Exact original experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home with registered images and fly graph.
    :type paths: Paths
    :param output: New attempt directory under FLYSTATE_HOME.
    :type output: Path
    :param case_id: N0, N1, or OFF in the frozen execution order.
    :type case_id: str
    :param cohort_path: Exact 40-image training-only selection.
    :type cohort_path: Path
    :param masks_path: T27 anatomical observation masks.
    :type masks_path: Path
    :param membership_path: Original training/validation split membership.
    :type membership_path: Path
    :param schedule_path: Machine-readable frozen timing and case schedule.
    :type schedule_path: Path
    :returns: Completed attempt ID, source digest, and numeric array summary.
    :rtype: dict[str, Any]
    :raises ValueError: If sources, timing, order, or resource limits differ.
    """
    if case_id not in CASES:
        raise ValueError('Unknown identity-access condition.')
    threads = cfg.brain.threads
    if threads is None:
        raise ValueError('Identity-access requires an explicit Numba thread count.')
    schedule = _read_json(path=schedule_path)
    _check_schedule(
        cfg=cfg,
        schedule=schedule,
        cohort_path=cohort_path,
        masks_path=masks_path,
        membership_path=membership_path,
    )
    cohort = _read_json(path=cohort_path)
    masks = _read_json(path=masks_path)
    membership = _read_json(path=membership_path)
    if (
        config_hash(cfg=cfg) != cohort['source_config_sha256']
        or cohort['source_pilot_masks_sha256'] != schedule['masks_sha256']
        or cohort['windows'] != schedule['windows']
        or set(masks['masks']) != set(schedule['masks_order'])
        or len(schedule['masks_order']) != len(masks['masks'])
        or len(cohort['samples']) != 2 * cfg.dataset.subset.n_identities
        or [item['label'] for item in cohort['samples']]
        != [label for label in range(cfg.dataset.subset.n_identities) for _ in range(2)]
    ):
        raise ValueError('Identity-access cohort or mask order differs from the protocol.')
    directory = output_path(path=output, paths=paths)
    if directory.name != case_id:
        raise ValueError('Attempt directory name must equal the frozen case identifier.')
    prior_cases = list(CASES)[: list(CASES).index(case_id)]
    elapsed_before = 0.0
    for prior in prior_cases:
        manifest = verify_attempt_inventory(directory=directory.parent / prior, paths=paths)
        if (
            manifest['status'] != 'completed'
            or manifest['parameters']['kind'] != 'identity_access'
            or manifest['parameters']['case'] != prior
            or manifest['parameters']['schedule_sha256'] != sha256_file(path=schedule_path)
        ):
            raise ValueError(f'Prior identity-access case is not complete: {prior}.')
        elapsed_before += float(manifest['elapsed_seconds'])
    if elapsed_before >= schedule['campaign_time_limit_seconds']:
        raise ValueError('Campaign wall-time budget was exhausted before this case.')
    case = CASES[case_id]
    parameters = {
        'kind': 'identity_access',
        'case': case_id,
        'hypothesis': 'Measure training-only identity structure in fixed fly responses.',
        'cohort_sha256': schedule['cohort_sha256'],
        'masks_sha256': schedule['masks_sha256'],
        'membership_sha256': schedule['membership_file_sha256'],
        'schedule_sha256': sha256_file(path=schedule_path),
        'noise_enabled': case['noise_enabled'],
        'episode_seed': case['episode_seed'],
    }
    started = perf_counter()
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as target:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        selected = _verify_cohort(
            cfg=cfg, paths=paths, prepared=prepared, cohort=cohort, membership=membership
        )
        populations = _verify_masks(paths=paths, document=masks)
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        original_threads = numba.get_num_threads()
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                warmup = EpisodeBrain(
                    brain_dir=paths.brain,
                    brain_cfg=cfg.brain,
                    readout_cfg=cfg.readout,
                    batch_size=1,
                    threads=threads,
                )
                rest = warmup.compute_rest_state(seed=cfg.seed)
                if case['noise_enabled']:
                    brain = warmup
                else:
                    del warmup
                    disabled = cfg.brain.noise.model_copy(update={'enabled': False})
                    brain = EpisodeBrain(
                        brain_dir=paths.brain,
                        brain_cfg=cfg.brain.model_copy(update={'noise': disabled}),
                        readout_cfg=cfg.readout,
                        batch_size=1,
                        threads=threads,
                    )
                encoder = SparseProjectionEncoder(
                    cfg=cfg.encoder,
                    window=cfg.episodes.window,
                    candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
                )
                recorded: dict[str, list[NDArray[Any]]] = {}
                blanks: dict[str, list[NDArray[Any]]] = {}
                noise_by_window: dict[int, tuple[NDArray[np.int64], NDArray[np.uint8]]] = {}
                for window in schedule['windows']:
                    key = _noise_identifier(window=window, case_id=case_id)
                    brain.begin(rest=rest, sample_ids=[key], seed=case['episode_seed'])
                    response = brain.run_recorded(
                        input_idx=encoder.input_idx,
                        currents=None,
                        stimulus_steps=schedule['stimulus_steps'],
                        recovery_steps=schedule['recovery_steps'],
                        populations=populations,
                        checkpoints=tuple(schedule['checkpoints']),
                    )
                    _save_response(
                        recorded=blanks, response=response, populations=schedule['masks_order']
                    )
                    noise_by_window[window] = (
                        response.noise_kicks[:, 0].copy(),
                        response.noise_digests[:, 0].copy(),
                    )
                rows: list[dict[str, Any]] = []
                for row_index, item in selected:
                    episode = builder.build(
                        sample=prepared.samples[row_index], image=prepared.images[row_index]
                    )
                    for window in schedule['windows']:
                        observation = episode.observations[window : window + 1]
                        positions = episode.positions[window : window + 1]
                        currents = encoder.encode(observations=observation, positions=positions)
                        key = _noise_identifier(window=window, case_id=case_id)
                        brain.begin(rest=rest, sample_ids=[key], seed=case['episode_seed'])
                        response = brain.run_recorded(
                            input_idx=encoder.input_idx,
                            currents=currents,
                            stimulus_steps=schedule['stimulus_steps'],
                            recovery_steps=schedule['recovery_steps'],
                            populations=populations,
                            checkpoints=tuple(schedule['checkpoints']),
                        )
                        noise_counts, noise_digests = noise_by_window[window]
                        if not np.array_equal(
                            response.noise_kicks[:, 0], noise_counts
                        ) or not np.array_equal(response.noise_digests[:, 0], noise_digests):
                            raise ValueError(
                                'Driven episode and blank have different noise indices.'
                            )
                        _save_response(
                            recorded=recorded,
                            response=response,
                            populations=schedule['masks_order'],
                        )
                        rows.append(
                            {
                                'sample_id': item['sample_id'],
                                'label': item['label'],
                                'window': window,
                                'pixel_sha256': hashlib.sha256(observation.tobytes()).hexdigest(),
                                'current_sha256': hashlib.sha256(currents.tobytes()).hexdigest(),
                                'current_l2': float(
                                    np.sqrt(
                                        np.sum(a=currents.astype(np.float64) ** 2, dtype=np.float64)
                                    )
                                ),
                                'current_nonzero': int(np.count_nonzero(a=currents)),
                            }
                        )
                        if (
                            psutil.Process().memory_info().rss > schedule['memory_limit_bytes']
                            or elapsed_before + perf_counter() - started
                            > schedule['campaign_time_limit_seconds']
                        ):
                            raise RuntimeError('Identity-access resource budget was exceeded.')
                numeric = {name: np.stack(arrays) for name, arrays in recorded.items()}
                blank_numeric = {name: np.stack(arrays) for name, arrays in blanks.items()}
                if not case['noise_enabled'] and (
                    np.any(numeric['noise_kicks']) or np.any(blank_numeric['noise_kicks'])
                ):
                    raise ValueError('Noise-off condition contained episode-noise kicks.')
                _write_arrays(path=target / 'responses.npz', arrays=numeric)
                _write_arrays(path=target / 'blanks.npz', arrays=blank_numeric)
                write_json(path=target / 'sample-windows.json', value=rows)
                report = {
                    'schema_version': 1,
                    'case': case_id,
                    'stimulus_steps': schedule['stimulus_steps'],
                    'recovery_steps': schedule['recovery_steps'],
                    'dt_s': cfg.brain.dt_s,
                    'checkpoints': schedule['checkpoints'],
                    'windows': schedule['windows'],
                    'populations': {name: len(indices) for name, indices in populations.items()},
                    'sample_windows': len(rows),
                    'blank_windows': len(schedule['windows']),
                    'noise_kicks': int(numeric['noise_kicks'].sum()),
                    'noise_digest_scheme': (
                        "SHA-256 of each step's ordered little-endian int64 neuron indices"
                    ),
                    'response_sha256': sha256_file(path=target / 'responses.npz'),
                    'blank_sha256': sha256_file(path=target / 'blanks.npz'),
                    'conclusion': 'Frozen identity responses recorded; paired analysis is pending.',
                }
                write_json(path=target / 'report.json', value=report)
        finally:
            numba.set_num_threads(n=original_threads)
    return {'status': 'completed', 'output': str(target), **report}

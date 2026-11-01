"""Record per-window population spike counts under a scaled encoder drive and decode identity."""

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from flystate.brain.benchmark import resolve_threads
from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.input_access import _features, _read_json, _verify_protocol
from flystate.diagnostics.input_loss_selection import _cross_validate
from flystate.diagnostics.temporal import _write_arrays
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj, stable_int
from flystate.settings import Paths, get_paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

OPTIC_LOBE_SAMPLE: int = 8000
ALL_WINDOWS: str = 'all_windows_counts'
LAST_WINDOW: str = 'last_window_counts'
FINAL_VOLTAGE: str = 'final_voltage'
REPRESENTATIONS: tuple[str, ...] = (ALL_WINDOWS, LAST_WINDOW, FINAL_VOLTAGE)
INPUT_REFERENCES: tuple[str, ...] = ('encoded_current', 'pixels')
INPUT_REFERENCE: str = 'input_reference'
COUNTS_PREFIX: str = 'counts_'
VOLTAGE_PREFIX: str = 'final_voltage_'
RESPONSES_FILE: str = 'responses.npz'
REPORT_FILE: str = 'report.json'
ISSUE: int = 64
STATUS: str = 'status'
COMPLETED: str = 'completed'
PARAMETERS: str = 'parameters'
COHORT_SHA256: str = 'cohort_sha256'
PARENT_SCHEDULE_SHA256: str = 'parent_schedule_sha256'
POPULATION_SIZES: str = 'population_sizes'
SAMPLES: str = 'samples'
SAMPLE_ID: str = 'sample_id'
LABEL: str = 'label'
ROLE: str = 'role'
ACCURACY: str = 'accuracy'
LOG_LOSS: str = 'log_loss'
FIT_ROLE: str = 'fit'
DECODE_RESOURCES: dict[str, int] = {'max_wall_minutes': 600, 'max_peak_rss_gib': 12}


@dataclass(frozen=True)
class SimulationJob:
    """One worker's share of a drive condition, passed to a spawned process."""

    config: dict[str, Any]
    home: str
    rows: list[int]
    amplitude_scale: float
    episode_seed: int


def population_indices(
    brain_file: Path, input_idx: NDArray[np.int64]
) -> dict[str, NDArray[np.int64]]:
    """Select the disjoint, predeclared T32 observation populations.

    Populations absent from the metadata are returned empty so that small synthetic brains remain
    usable. The optic-lobe sample takes the lowest ``(stable_int(str(bodyId)), bodyId)`` ranks, the
    same deterministic rule as the T27 masks.

    :param brain_file: flybrain ``brain.npz`` with ``ids``, ``superclass`` and ``cell_type``.
    :type brain_file: Path
    :param input_idx: Encoder-driven neuron indices, shape (I,), int64.
    :type input_idx: NDArray[np.int64]
    :returns: Sorted unique int64 neuron indices per population name, each shape (K,).
    :rtype: dict[str, NDArray[np.int64]]
    """
    with np.load(file=brain_file, allow_pickle=False) as meta:
        superclass = meta['superclass'].astype(str)
        cell_type = meta['cell_type'].astype(str)
        ids = meta['ids']
    driven = np.zeros(shape=len(superclass), dtype=np.bool_)
    driven[input_idx] = True
    kenyon = np.char.startswith(cell_type, 'KC')
    optic_lobe = np.flatnonzero(superclass == 'ol_intrinsic')
    ranked = sorted(
        optic_lobe, key=lambda index: (stable_int(key=str(ids[index])), int(ids[index]))
    )
    selected = {
        'driven': np.flatnonzero(driven),
        'visual_projection_other': np.flatnonzero((superclass == 'visual_projection') & ~driven),
        'central_brain': np.flatnonzero((superclass == 'cb_intrinsic') & ~kenyon & ~driven),
        'kenyon': np.flatnonzero(kenyon & ~driven),
        'descending': np.flatnonzero((superclass == 'descending_neuron') & ~driven),
        'optic_lobe_sample': np.sort(np.asarray(ranked[:OPTIC_LOBE_SAMPLE], dtype=np.int64)),
        'vnc': np.flatnonzero((superclass == 'vnc_intrinsic') & ~driven),
    }
    return {name: indices.astype(np.int64) for name, indices in selected.items()}


def _simulate_rows(job: SimulationJob) -> dict[str, NDArray[Any]]:
    """Simulate persistent episodes for one worker and return its per-population arrays.

    Runs in a separate process so that each worker owns one simulator and its Numba threads. The
    per-episode noise stream depends only on the episode seed and the sample identifier, so the
    result does not depend on how rows are assigned to workers.

    :param job: Config dict (noise already set), data home, row indices, scale and episode seed.
    :type job: SimulationJob
    :returns: ``counts_<name>`` uint8 (R,T,K) and ``final_voltage_<name>`` float32 (R,K).
    :rtype: dict[str, NDArray[Any]]
    """
    cfg = ExperimentConfig.model_validate(obj=job.config)
    paths = get_paths(home=Path(job.home))
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    brain = EpisodeBrain(
        brain_dir=paths.brain,
        brain_cfg=cfg.brain,
        readout_cfg=cfg.readout,
        batch_size=1,
        threads=resolve_threads(paths=paths, configured=cfg.brain.threads),
    )
    encoder = SparseProjectionEncoder(
        cfg=cfg.encoder,
        window=cfg.episodes.window,
        candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
    )
    populations = {
        name: indices
        for name, indices in population_indices(
            brain_file=paths.brain / 'brain.npz', input_idx=encoder.input_idx
        ).items()
        if indices.size
    }
    rest = brain.compute_rest_state(seed=cfg.seed)
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    windows, steps = cfg.episodes.steps, cfg.brain.steps_per_observation
    arrays: dict[str, NDArray[Any]] = {}
    for name, indices in populations.items():
        arrays[f'{COUNTS_PREFIX}{name}'] = np.zeros(
            shape=(len(job.rows), windows, len(indices)), dtype=np.uint8
        )
        arrays[f'{VOLTAGE_PREFIX}{name}'] = np.zeros(
            shape=(len(job.rows), len(indices)), dtype=np.float32
        )
    scale = np.float32(job.amplitude_scale)
    for offset, row in enumerate(job.rows):
        episode = builder.build(sample=prepared.samples[row], image=prepared.images[row])
        brain.begin(rest=rest, sample_ids=[episode.sample_id], seed=job.episode_seed)
        for window in range(windows):
            currents = encoder.encode(
                observations=episode.observations[window][None],
                positions=episode.positions[window][None],
            )
            response = brain.run_recorded(
                input_idx=encoder.input_idx,
                currents=(currents * scale).astype(np.float32),
                stimulus_steps=steps,
                recovery_steps=0,
                populations=populations,
                checkpoints=(0, steps),
            )
            for name in populations:
                arrays[f'{COUNTS_PREFIX}{name}'][offset, window] = response.spike_counts[name][
                    -1, 0
                ]
                if window == windows - 1:
                    arrays[f'{VOLTAGE_PREFIX}{name}'][offset] = response.voltages[name][-1, 0]
    return arrays


def simulate_counts(
    cfg: ExperimentConfig,
    paths: Paths,
    rows: list[int],
    amplitude_scale: float,
    noise_enabled: bool,
    episode_seed: int,
    workers: int,
) -> dict[str, NDArray[Any]]:
    """Simulate episodes in parallel processes and merge rows in the requested order.

    :param cfg: Original experiment configuration; only noise is overridden.
    :type cfg: ExperimentConfig
    :param paths: Data home with brain files and the prepared dataset cache.
    :type paths: Paths
    :param rows: Prepared-dataset row indices in output order, length N.
    :type rows: list[int]
    :param amplitude_scale: Nonnegative multiplier of the encoder current, dimensionless.
    :type amplitude_scale: float
    :param noise_enabled: Whether per-episode background noise is applied.
    :type noise_enabled: bool
    :param episode_seed: Nonnegative episode-noise seed.
    :type episode_seed: int
    :param workers: Positive number of simulator processes.
    :type workers: int
    :returns: ``counts_<name>`` uint8 (N,T,K) and ``final_voltage_<name>`` float32 (N,K).
    :rtype: dict[str, NDArray[Any]]
    :raises ValueError: If the scale, seed, rows, or worker count are invalid.
    """
    if not math.isfinite(amplitude_scale) or amplitude_scale < 0 or episode_seed < 0:
        raise ValueError('Amplitude scale and episode seed must be finite and nonnegative.')
    if not rows or workers < 1:
        raise ValueError('At least one row and one worker are required.')
    # Build or verify the shared image cache once, before workers read it concurrently.
    prepare_dataset(cfg=cfg, paths=paths)
    config = cfg.model_dump(mode='json')
    config['brain']['noise']['enabled'] = noise_enabled
    shards = [list(range(start, len(rows), workers)) for start in range(min(workers, len(rows)))]
    jobs = [
        SimulationJob(
            config=config,
            home=str(paths.home),
            rows=[rows[position] for position in shard],
            amplitude_scale=amplitude_scale,
            episode_seed=episode_seed,
        )
        for shard in shards
    ]
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context('spawn')) as pool:
        parts = list(pool.map(_simulate_rows, jobs))
    merged: dict[str, NDArray[Any]] = {}
    for key in parts[0]:
        shape = (len(rows), *parts[0][key].shape[1:])
        merged[key] = np.empty(shape=shape, dtype=parts[0][key].dtype)
        for shard, part in zip(shards, parts, strict=True):
            merged[key][shard] = part[key]
    return merged


def record_drive(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    cohort_path: Path,
    parent_schedule_path: Path,
    membership_path: Path,
    amplitude_scale: float,
    noise_enabled: bool,
    episode_seed: int,
    workers: int,
) -> dict[str, Any]:
    """Record one T32 drive condition for the 280 frozen training photographs.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param cohort_path: T29 committed 280-image cohort document.
    :type cohort_path: Path
    :param parent_schedule_path: T29 committed exact CV-fold document.
    :type parent_schedule_path: Path
    :param membership_path: Archived original split membership.
    :type membership_path: Path
    :param amplitude_scale: Encoder current multiplier; 0 records a paired blank.
    :type amplitude_scale: float
    :param noise_enabled: Whether per-episode background noise is applied.
    :type noise_enabled: bool
    :param episode_seed: Episode-noise seed; the warmup always uses the config seed.
    :type episode_seed: int
    :param workers: Simulator processes, each using the configured Numba threads.
    :type workers: int
    :returns: Condition summary with mean spikes per window per population.
    :rtype: dict[str, Any]
    """
    parameters = {
        'kind': 'drive_sweep_record',
        'issue': ISSUE,
        'amplitude_scale': amplitude_scale,
        'max_kick_v_per_step': amplitude_scale * cfg.encoder.amplitude,
        'noise_enabled': noise_enabled,
        'episode_seed': episode_seed,
        'warmup_seed': cfg.seed,
        'workers': workers,
        'numba_threads_per_worker': cfg.brain.threads,
        COHORT_SHA256: sha256_file(path=cohort_path),
        PARENT_SCHEDULE_SHA256: sha256_file(path=parent_schedule_path),
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        cohort = _read_json(path=cohort_path)
        rows = _verify_protocol(
            cfg=cfg,
            paths=paths,
            prepared=prepared,
            cohort=cohort,
            schedule=_read_json(path=parent_schedule_path),
            membership=_read_json(path=membership_path),
            membership_path=membership_path,
            cohort_path=cohort_path,
        )
        started = perf_counter()
        with threadpool_limits(limits=1, user_api='blas'):
            arrays = simulate_counts(
                cfg=cfg,
                paths=paths,
                rows=rows,
                amplitude_scale=amplitude_scale,
                noise_enabled=noise_enabled,
                episode_seed=episode_seed,
                workers=workers,
            )
        elapsed = perf_counter() - started
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('Recorded population arrays contain nonfinite values.')
        _write_arrays(
            path=directory / RESPONSES_FILE,
            arrays={
                'labels': np.asarray([item[LABEL] for item in cohort[SAMPLES]], dtype=np.int64),
                **arrays,
            },
        )
        populations = {
            key.removeprefix(COUNTS_PREFIX): value
            for key, value in arrays.items()
            if key.startswith(COUNTS_PREFIX)
        }
        summary = {
            PARAMETERS: parameters,
            'simulation_seconds': elapsed,
            'episodes': len(rows),
            'sample_ids': [item[SAMPLE_ID] for item in cohort[SAMPLES]],
            'roles': [item[ROLE] for item in cohort[SAMPLES]],
            POPULATION_SIZES: {name: value.shape[2] for name, value in populations.items()},
            'mean_spikes_per_window': {
                name: float(value.sum(axis=2, dtype=np.int64).mean())
                for name, value in populations.items()
            },
            'responses_sha256': sha256_file(path=directory / RESPONSES_FILE),
        }
        write_json(path=directory / REPORT_FILE, value=summary)
    return summary


def decode_drive(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    recordings: list[Path],
    cohort_path: Path,
    parent_schedule_path: Path,
    membership_path: Path,
    selection_schedule_path: Path,
) -> dict[str, Any]:
    """Decode identity from recorded conditions with the exact T30 fit-only OOF procedure.

    Only the 200 ``fit`` photographs enter decoding. The encoded-current and pixel references are
    recomputed on the same photographs and folds. A nonconverged candidate is recorded as a failed
    case rather than substituted.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param recordings: Completed ``record_drive`` attempt directories.
    :type recordings: list[Path]
    :param cohort_path: T29 committed 280-image cohort document.
    :type cohort_path: Path
    :param parent_schedule_path: T29 committed exact CV-fold document.
    :type parent_schedule_path: Path
    :param membership_path: Archived original split membership.
    :type membership_path: Path
    :param selection_schedule_path: T30 committed C grid, PCA and convergence schedule.
    :type selection_schedule_path: Path
    :returns: Per-condition, population and representation OOF metrics.
    :rtype: dict[str, Any]
    """
    parameters = {
        'kind': 'drive_sweep_decode',
        'issue': ISSUE,
        'recordings': [str(path) for path in recordings],
        COHORT_SHA256: sha256_file(path=cohort_path),
        PARENT_SCHEDULE_SHA256: sha256_file(path=parent_schedule_path),
        'selection_schedule_sha256': sha256_file(path=selection_schedule_path),
        'representations': list(REPRESENTATIONS),
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        started = perf_counter()
        cohort = _read_json(path=cohort_path)
        parent_schedule = _read_json(path=parent_schedule_path)
        schedule = _read_json(path=selection_schedule_path)
        schedule['resources'] = DECODE_RESOURCES
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        rows = _verify_protocol(
            cfg=cfg,
            paths=paths,
            prepared=prepared,
            cohort=cohort,
            schedule=parent_schedule,
            membership=_read_json(path=membership_path),
            membership_path=membership_path,
            cohort_path=cohort_path,
        )
        fit_items = [item for item in cohort[SAMPLES] if item[ROLE] == FIT_ROLE]
        fit = np.array([item[ROLE] == FIT_ROLE for item in cohort[SAMPLES]], dtype=np.bool_)
        ids = [item[SAMPLE_ID] for item in fit_items]
        labels = np.asarray([item[LABEL] for item in fit_items], dtype=np.int64)
        references, _ = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        cases: list[tuple[str, str, str, NDArray[np.float32]]] = []
        for name in INPUT_REFERENCES:
            values = references[name][fit]
            window_width = values.shape[1] // cfg.episodes.steps
            cases.append((INPUT_REFERENCE, name, ALL_WINDOWS, values))
            cases.append((INPUT_REFERENCE, name, LAST_WINDOW, values[:, -window_width:]))
        for recording in recordings:
            manifest = verify_attempt_inventory(directory=recording, paths=paths)
            source = output_path(path=recording, paths=paths)
            report = _read_json(path=source / REPORT_FILE)
            if (
                manifest[STATUS] != COMPLETED
                or report[PARAMETERS][COHORT_SHA256] != parameters[COHORT_SHA256]
            ):
                raise ValueError(f'Recording is incomplete or from another cohort: {recording}.')
            condition = source.name
            with np.load(file=source / RESPONSES_FILE, allow_pickle=False) as data:
                if [
                    sample_id
                    for sample_id, keep in zip(report['sample_ids'], fit, strict=True)
                    if keep
                ] != ids:
                    raise ValueError(f'Recording sample order differs: {recording}.')
                for name in report[POPULATION_SIZES]:
                    counts = data[f'{COUNTS_PREFIX}{name}'][fit].astype(np.float32)
                    cases.append((condition, name, ALL_WINDOWS, counts.reshape(len(ids), -1)))
                    cases.append((condition, name, LAST_WINDOW, counts[:, -1]))
                    voltage = data[f'{VOLTAGE_PREFIX}{name}'][fit].astype(np.float32)
                    cases.append((condition, name, FINAL_VOLTAGE, voltage))
        metrics: list[dict[str, Any]] = []
        oof_rows: list[dict[str, Any]] = []
        for condition, population, representation, x in cases:
            label = f'{condition}/{population}/{representation}'
            row: dict[str, Any] = {
                'condition': condition,
                'population': population,
                'representation': representation,
                'features': int(x.shape[1]),
                'nonconstant_features': int(np.count_nonzero(x.std(axis=0) > 0)),
                'feature_sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            }
            try:
                case_oof, _, summaries, selected = _cross_validate(
                    x=x,
                    labels=labels,
                    ids=ids,
                    schedule=schedule,
                    parent_folds=parent_schedule['cv'],
                    representation=label,
                    started=started,
                )
            except ConvergenceWarning as error:
                metrics.append({**row, STATUS: 'nonconverged', 'error': str(error)})
                continue
            chosen = next(item for item in summaries if item['C'] == selected)
            metrics.append(
                {
                    **row,
                    STATUS: COMPLETED,
                    'C': selected,
                    'top1_correct': round(chosen[ACCURACY] * len(ids)),
                    'top1': chosen[ACCURACY],
                    'top5': chosen['top5_accuracy'],
                    LOG_LOSS: chosen[LOG_LOSS],
                    'candidates': [
                        {key: item[key] for key in ('C', ACCURACY, LOG_LOSS)} for item in summaries
                    ],
                }
            )
            oof_rows.extend(item for item in case_oof if item['C'] == selected)
        write_table(path=directory / 'oof-predictions.parquet', rows=oof_rows)
        report = {
            PARAMETERS: parameters,
            'chance_top1': 1 / 20,
            'delivery_threshold_top1_correct': 46,
            'delivery_threshold_log_loss': math.log(20),
            'metrics': metrics,
            'metrics_sha256': sha256_obj(obj=metrics),
            'validation_test_reserve_scored': False,
            'prior_queries_scored': False,
        }
        write_json(path=directory / REPORT_FILE, value=report)
    return {
        'output': str(directory),
        'cases': len(metrics),
        COMPLETED: sum(item[STATUS] == COMPLETED for item in metrics),
    }

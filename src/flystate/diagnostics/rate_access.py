"""Record identity-bearing population states of the MaleCNS graph under graded dynamics (T33)."""

import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.benchmark import resolve_threads
from flystate.brain.rate import RateReservoir
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.drive_sweep import (
    ARRAY_PREFIXES,
    COHORT_SHA256,
    LABEL,
    PARAMETERS,
    PARENT_SCHEDULE_SHA256,
    POPULATION_SIZES,
    REPORT_FILE,
    RESPONSES_FILE,
    ROLE,
    SAMPLE_ID,
    SAMPLES,
    STATE_PREFIX,
    population_indices,
)
from flystate.diagnostics.input_access import _features, _read_json, _verify_protocol
from flystate.diagnostics.matched_neural import _paired_analysis
from flystate.diagnostics.temporal import _write_arrays
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, stable_int
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table

ENCODED_CURRENT: str = 'encoded_current'
BRAIN_METADATA: str = 'brain.npz'
MEMORY_BOOTSTRAP_NAMESPACE: str = 't33-memory-identity-bootstrap'


def simulate_states(
    reservoir: RateReservoir,
    input_idx: NDArray[np.int64],
    inputs: NDArray[np.float32],
    populations: dict[str, NDArray[np.int64]],
    steps_per_window: int,
    reset_each_window: bool,
) -> dict[str, NDArray[np.float32]]:
    """Run all episodes together and keep each population's state at the end of every window.

    :param reservoir: Rate reservoir whose batch size equals the episode count B.
    :type reservoir: RateReservoir
    :param input_idx: Driven neuron indices, shape (I,), int64.
    :type input_idx: NDArray[np.int64]
    :param inputs: Per-window input held constant within the window, shape (B,T,I), float32.
    :type inputs: NDArray[np.float32]
    :param populations: Named sorted neuron indices, each shape (K,).
    :type populations: dict[str, NDArray[np.int64]]
    :param steps_per_window: Positive number of synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Zero the state before every window (the no-memory control).
    :type reset_each_window: bool
    :returns: ``state_<name>`` arrays of shape (B,T,K), float32.
    :rtype: dict[str, NDArray[np.float32]]
    :raises ValueError: If the input shape or step count is invalid.
    """
    episodes, windows, driven = inputs.shape
    if driven != len(input_idx) or episodes != reservoir.state.shape[1] or steps_per_window < 1:
        raise ValueError('Inputs must be (B,T,I) for the reservoir batch and driven neurons.')
    states = {
        f'{STATE_PREFIX}{name}': np.empty(shape=(episodes, windows, len(indices)), dtype=np.float32)
        for name, indices in populations.items()
    }
    reservoir.reset()
    for window in range(windows):
        if reset_each_window:
            reservoir.reset()
        drive = np.ascontiguousarray(inputs[:, window].T)
        for _ in range(steps_per_window):
            reservoir.step(input_idx=input_idx, inputs=drive)
        for name, indices in populations.items():
            states[f'{STATE_PREFIX}{name}'][:, window] = reservoir.state[indices].T
    return states


def record_rate(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    cohort_path: Path,
    parent_schedule_path: Path,
    membership_path: Path,
    gain: float,
    leak: float,
    input_scale: float,
    steps_per_window: int,
    reset_each_window: bool,
) -> dict[str, Any]:
    """Record one graded-dynamics condition for the 280 frozen training photographs.

    The input is the original sparse encoder current multiplied by ``input_scale``; the driven
    neurons, windows, cohort and observed populations are those of T32.

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
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: Update fraction in (0, 1].
    :type leak: float
    :param input_scale: Multiplier of the encoded current, dimensionless.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Whether the state is zeroed before every window.
    :type reset_each_window: bool
    :returns: Condition summary with state statistics per population.
    :rtype: dict[str, Any]
    :raises ValueError: If the input scale is invalid or a state is not finite.
    """
    if not math.isfinite(input_scale) or input_scale < 0:
        raise ValueError('Input scale must be finite and nonnegative.')
    threads = resolve_threads(paths=paths, configured=cfg.brain.threads)
    parameters = {
        'kind': 'rate_access_record',
        'issue': 65,
        'model': 'x <- (1 - leak) x + leak tanh(gain W_eff x + input_scale * encoded_current)',
        'gain': gain,
        'leak': leak,
        'input_scale': input_scale,
        'steps_per_window': steps_per_window,
        'reset_each_window': reset_each_window,
        'noise': False,
        'numba_threads': threads,
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
        features, provenance = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        currents = features[ENCODED_CURRENT].reshape(len(rows), cfg.episodes.steps, -1)
        numba.set_num_threads(threads)
        reservoir = RateReservoir(brain_dir=paths.brain, gain=gain, leak=leak, batch_size=len(rows))
        with np.load(file=paths.brain / BRAIN_METADATA, allow_pickle=False) as meta:
            candidates = np.flatnonzero(meta['superclass'] == cfg.encoder.target_population)
        encoder = SparseProjectionEncoder(
            cfg=cfg.encoder, window=cfg.episodes.window, candidate_neurons=candidates
        )
        populations = {
            name: indices
            for name, indices in population_indices(
                brain_file=paths.brain / BRAIN_METADATA, input_idx=encoder.input_idx
            ).items()
            if indices.size
        }
        started = perf_counter()
        with threadpool_limits(limits=1, user_api='blas'):
            states = simulate_states(
                reservoir=reservoir,
                input_idx=encoder.input_idx,
                inputs=(currents * np.float32(input_scale)).astype(np.float32),
                populations=populations,
                steps_per_window=steps_per_window,
                reset_each_window=reset_each_window,
            )
        elapsed = perf_counter() - started
        if not all(np.isfinite(value).all() for value in states.values()):
            raise ValueError('Recorded rate states contain nonfinite values.')
        _write_arrays(
            path=directory / RESPONSES_FILE,
            arrays={
                'labels': np.asarray([item[LABEL] for item in cohort[SAMPLES]], dtype=np.int64),
                **states,
            },
        )
        summary = {
            PARAMETERS: parameters,
            ARRAY_PREFIXES: {'window': STATE_PREFIX},
            'simulation_seconds': elapsed,
            'episodes': len(rows),
            'edges': reservoir.edges,
            'encoder_input_sha256': provenance['feature_sha256'][ENCODED_CURRENT],
            'sample_ids': [item[SAMPLE_ID] for item in cohort[SAMPLES]],
            'roles': [item[ROLE] for item in cohort[SAMPLES]],
            POPULATION_SIZES: {name: len(indices) for name, indices in populations.items()},
            'final_state_mean_abs': {
                name: float(np.abs(states[f'{STATE_PREFIX}{name}'][:, -1]).mean())
                for name in populations
            },
            'final_state_saturated_fraction': {
                name: float(np.mean(np.abs(states[f'{STATE_PREFIX}{name}'][:, -1]) > 0.99))
                for name in populations
            },
            'responses_sha256': sha256_file(path=directory / RESPONSES_FILE),
        }
        write_json(path=directory / REPORT_FILE, value=summary)
    return summary


def analyze_memory(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    decode: Path,
    settings: list[str],
    populations: list[str],
    representation: str,
    minimum_gain_pp: float,
) -> dict[str, Any]:
    """Compare persistent with reset last-window states, paired by photograph, for every setting.

    Condition names follow ``<setting>-persistent`` and ``<setting>-reset``. The interval uses the
    T31 identity-cluster bootstrap with 10,000 seeded draws, and the gate uses the Bonferroni level
    ``1 - 0.05 / len(settings)``.

    :param cfg: Original experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param decode: Completed ``drive-decode`` attempt containing every named condition.
    :type decode: Path
    :param settings: Dynamics setting names, each with a persistent and a reset condition.
    :type settings: list[str]
    :param populations: Populations to test, for example the central brain and descending neurons.
    :type populations: list[str]
    :param representation: Decoded representation compared, for example the last-window state.
    :type representation: str
    :param minimum_gain_pp: Required persistent-minus-reset accuracy, percentage points.
    :type minimum_gain_pp: float
    :returns: Per-setting memory comparisons and the gate outcome per population.
    :rtype: dict[str, Any]
    """
    parameters = {
        'kind': 'rate_access_memory',
        'issue': 65,
        'decode': str(decode),
        'settings': settings,
        'populations': populations,
        'representation': representation,
        'minimum_gain_pp': minimum_gain_pp,
        'bootstrap_seed_namespace': MEMORY_BOOTSTRAP_NAMESPACE,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        manifest = verify_attempt_inventory(directory=decode, paths=paths)
        source = output_path(path=decode, paths=paths)
        if manifest['status'] != 'completed':
            raise ValueError('The decode attempt is not completed.')
        metrics = {
            (row['condition'], row['population'], row['representation']): row
            for row in _read_json(path=source / REPORT_FILE)['metrics']
        }
        oof = read_table(path=source / 'oof-predictions.parquet')
        rows_by_label: dict[str, list[dict[str, Any]]] = {}
        for row in oof:
            rows_by_label.setdefault(row['representation'], []).append(row)
        generator = np.random.default_rng(
            seed=np.random.SeedSequence(
                entropy=[cfg.seed, stable_int(key=MEMORY_BOOTSTRAP_NAMESPACE)]
            )
        )
        draws = generator.integers(low=0, high=20, size=(10000, 20), dtype=np.int64)
        level = 0.05 / len(settings)
        comparisons: list[dict[str, Any]] = []
        for population in populations:
            for setting in settings:
                persistent, reset = (f'{setting}-{kind}' for kind in ('persistent', 'reset'))
                keys = [(name, population, representation) for name in (reset, persistent)]
                summary, _, distribution = _paired_analysis(
                    case=f'{setting}/{population}',
                    input_rows=rows_by_label['/'.join(keys[0])],
                    neural_rows=rows_by_label['/'.join(keys[1])],
                    input_metrics={
                        'accuracy': metrics[keys[0]]['top1'],
                        'log_loss': metrics[keys[0]]['log_loss'],
                    },
                    neural_metrics={
                        'accuracy': metrics[keys[1]]['top1'],
                        'log_loss': metrics[keys[1]]['log_loss'],
                    },
                    bootstrap_draws=draws,
                )
                values = np.asarray([row['neural_minus_input_accuracy_pp'] for row in distribution])
                low, high = np.percentile(values, q=[50 * level, 100 - 50 * level])
                gain = summary['neural_minus_input_accuracy_pp']
                comparisons.append(
                    {
                        'setting': setting,
                        'population': population,
                        'reset_correct': round(summary['input_accuracy'] * 200),
                        'persistent_correct': round(summary['neural_accuracy'] * 200),
                        'persistent_minus_reset_pp': gain,
                        'interval_95_pp': [
                            summary['cluster_interval_low_pp'],
                            summary['cluster_interval_high_pp'],
                        ],
                        'interval_bonferroni_pp': [float(low), float(high)],
                        'memory_supported': bool(gain >= minimum_gain_pp and low > 0),
                        'error_overlap': summary['error_overlap'],
                    }
                )
        report = {
            PARAMETERS: parameters,
            'bonferroni_level': level,
            'comparisons': comparisons,
            'gate': {
                population: any(
                    row['memory_supported']
                    for row in comparisons
                    if row['population'] == population
                )
                for population in populations
            },
        }
        write_json(path=directory / REPORT_FILE, value=report)
    return report['gate']

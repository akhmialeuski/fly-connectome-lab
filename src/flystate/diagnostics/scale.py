"""The confirmed graded MaleCNS model at 100 identities, over encoder seeds and blank delays (T38).

Recording reuses the T35 per-cohort recorder with photograph batches, so a 2,000-photograph cohort
fits in memory, and with blank windows after the last glimpse, so forgetting can be read from the
same run. Evaluation reuses the T35 evaluation. This module adds the recorder and the pooled
summary: recognition and memory per recording, their spread across encoder seeds, and the
forgetting curve with its half-retention delay.
"""

from pathlib import Path
from typing import Any

import numba
import numpy as np

from flystate.brain.benchmark import resolve_threads
from flystate.brain.rate import RateReservoir
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.confirmation import (
    CHANCE_ALPHA,
    COMPARISONS,
    DELAY_INFIX,
    ISSUE_KEY,
    KIND,
    SCORES,
    driven_neurons,
    record_cohort,
)
from flystate.diagnostics.drive_sweep import PARAMETERS, REPORT_FILE
from flystate.diagnostics.input_access import _read_json
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json

ISSUE: int = 75
KEEP: tuple[str, ...] = ('central_brain', 'descending')
PRIMARY: str = 'central_brain'
PERSISTENT: str = 'persistent'
RESET: str = 'reset'
MINIMUM_MEMORY_GAIN_PP: float = 10.0
HALF: float = 0.5
PERCENT: float = 100.0
ACCURACY: str = 'accuracy'
COMPLETED: str = 'completed'
DIFFERENCE: str = 'a_minus_b_pp'
ACCURACY_PCT: str = 'accuracy_pct'
BINOMIAL_P: str = 'binomial_p_above_chance'
BLANK_DELAYS: str = 'blank_delays'
CHANCE: str = 'chance'
CONFIRMATION: str = 'confirmation'
DEVELOPMENT: str = 'development'
HELD_OUT: str = 'held_out'
MEMORY_GAIN: str = 'memory_gain_pp'
MEMORY_PASSES: str = 'memory_passes'
RECOGNITION_PASSES: str = 'recognition_passes'


def record_scale(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    gain: float,
    leak: float,
    driven_leak: float,
    input_scale: float,
    steps_per_window: int,
    reset_each_window: bool,
    delays: tuple[int, ...],
    batch_size: int,
    threads: int | None,
) -> dict[str, Any]:
    """Record the final central-brain and descending states of a cohort, in photograph batches.

    :param cfg: Cohort configuration, including its encoder seed.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: Update fraction of the non-driven neurons.
    :type leak: float
    :param driven_leak: Update fraction of the driven neurons.
    :type driven_leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Whether the state is zeroed before every window.
    :type reset_each_window: bool
    :param delays: Numbers of blank windows after the last glimpse at which states are kept.
    :type delays: tuple[int, ...]
    :param batch_size: Photographs simulated together; must divide the cohort.
    :type batch_size: int
    :param threads: Numba threads, or the configuration's setting.
    :type threads: Optional[int]
    :returns: Recording summary.
    :rtype: dict[str, Any]
    """
    resolved = resolve_threads(paths=paths, configured=threads or cfg.brain.threads)
    parameters = {
        KIND: 'scale_record',
        ISSUE_KEY: ISSUE,
        'config_sha256': config_hash(cfg=cfg),
        'encoder_seed': cfg.encoder.seed,
        'selection_seed': cfg.dataset.subset.selection_seed,
        'gain': gain,
        'leak': leak,
        'driven_leak': driven_leak,
        'input_scale': input_scale,
        'steps_per_window': steps_per_window,
        'reset_each_window': reset_each_window,
        BLANK_DELAYS: list(delays),
        'batch_size': batch_size,
        'numba_threads': resolved,
        'populations': list(KEEP),
        'graph': 'fly',
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        input_idx = driven_neurons(cfg=cfg, paths=paths)
        numba.set_num_threads(resolved)
        reservoir = RateReservoir(
            brain_dir=paths.brain,
            gain=gain,
            leak=leak,
            batch_size=batch_size,
            leak_overrides={driven_leak: input_idx},
        )
        return record_cohort(
            cfg=cfg,
            paths=paths,
            directory=directory,
            reservoir=reservoir,
            input_idx=input_idx,
            input_scale=input_scale,
            steps_per_window=steps_per_window,
            reset_each_window=reset_each_window,
            parameters=parameters,
            keep=KEEP,
            delays=delays,
        )


def half_retention(delays: list[int], accuracies: list[float], chance: float) -> float | None:
    """Return the blank delay at which accuracy above chance falls to half its delay-0 value.

    :param delays: Increasing blank delays, the first being 0, in windows.
    :type delays: list[int]
    :param accuracies: Held-out accuracy at each delay, fractions.
    :type accuracies: list[float]
    :param chance: Uniform-choice accuracy, fraction.
    :type chance: float
    :returns: Linearly interpolated delay in windows, or none if retention never halves.
    :rtype: Optional[float]
    """
    excess = [(accuracy - chance) / (accuracies[0] - chance) for accuracy in accuracies]
    for index in range(1, len(delays)):
        if excess[index] <= HALF:
            before, after = excess[index - 1], excess[index]
            fraction = (before - HALF) / (before - after)
            return delays[index - 1] + fraction * (delays[index] - delays[index - 1])
    return None


def _summary(report: dict[str, Any], delays: list[int]) -> dict[str, Any]:
    """Extract recognition, memory and forgetting of one evaluated cohort.

    :param report: T35-style evaluation report with persistent and reset cases.
    :type report: dict[str, Any]
    :param delays: Blank delays recorded for the persistent state.
    :type delays: list[int]
    :returns: Accuracy, binomial p, memory gain with interval, curve and half-retention.
    :rtype: dict[str, Any]
    """
    scores = report[SCORES]
    persistent = scores[f'{PERSISTENT}/{PRIMARY}']
    curve = [
        scores[f'{PERSISTENT}/{PRIMARY}' + ('' if d == 0 else f'{DELAY_INFIX}{d}')][ACCURACY]
        for d in delays
    ]
    memory = next(
        item
        for item in report[COMPARISONS]
        if item['a'] == f'{PERSISTENT}/{PRIMARY}' and item['b'] == f'{RESET}/{PRIMARY}'
    )
    low, high = memory['identity_cluster_95_pp']
    return {
        ACCURACY_PCT: persistent[ACCURACY] * PERCENT,
        HELD_OUT: persistent[HELD_OUT],
        'chance_pct': persistent[CHANCE] * PERCENT,
        BINOMIAL_P: persistent[BINOMIAL_P],
        RECOGNITION_PASSES: persistent[BINOMIAL_P] < CHANCE_ALPHA,
        'reset_accuracy_pct': scores[f'{RESET}/{PRIMARY}'][ACCURACY] * PERCENT,
        MEMORY_GAIN: memory[DIFFERENCE],
        'memory_interval_95_pp': [low, high],
        'memory_mcnemar': memory['mcnemar'],
        MEMORY_PASSES: memory[DIFFERENCE] >= MINIMUM_MEMORY_GAIN_PP and low > 0,
        'forgetting_curve_pct': dict(
            zip(map(str, delays), (a * PERCENT for a in curve), strict=True)
        ),
        'half_retention_windows': half_retention(
            delays=delays, accuracies=curve, chance=persistent[CHANCE]
        ),
    }


def analyze_scale(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    development: dict[str, Path],
    confirmation: Path,
    delays: list[int],
) -> dict[str, Any]:
    """Apply the frozen T38 rules to the development seeds and the single confirmation.

    :param cfg: Development configuration, for the attempt record.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New immutable attempt directory.
    :type output: Path
    :param development: Encoder-seed label to completed development evaluation.
    :type development: dict[str, Path]
    :param confirmation: Completed confirmation evaluation.
    :type confirmation: Path
    :param delays: Blank delays recorded for the persistent state, starting with 0.
    :type delays: list[int]
    :returns: Per-seed and confirmation summaries, the seed spread and the decisions.
    :rtype: dict[str, Any]
    :raises ValueError: If an evaluation is incomplete or the delays do not start at 0.
    """
    if not delays or delays[0] != 0 or sorted(delays) != delays:
        raise ValueError('Delays must be increasing and start with 0.')
    parameters = {
        KIND: 'scale_analysis',
        ISSUE_KEY: ISSUE,
        DEVELOPMENT: {name: str(path) for name, path in development.items()},
        CONFIRMATION: str(confirmation),
        BLANK_DELAYS: delays,
        'minimum_memory_gain_pp': MINIMUM_MEMORY_GAIN_PP,
        'chance_alpha': CHANCE_ALPHA,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        reports = {}
        for name, path in {**development, CONFIRMATION: confirmation}.items():
            if verify_attempt_inventory(directory=path, paths=paths)['status'] != COMPLETED:
                raise ValueError(f'The evaluation is not completed: {path}.')
            reports[name] = _read_json(path=output_path(path=path, paths=paths) / REPORT_FILE)
        seeds = {name: _summary(report=reports[name], delays=delays) for name in development}
        confirmed = _summary(report=reports[CONFIRMATION], delays=delays)
        spread = {
            key: {
                'mean': float(np.mean(values)),
                'standard_deviation': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                'minimum': float(np.min(values)),
                'maximum': float(np.max(values)),
            }
            for key, values in (
                (ACCURACY_PCT, [s[ACCURACY_PCT] for s in seeds.values()]),
                (MEMORY_GAIN, [s[MEMORY_GAIN] for s in seeds.values()]),
            )
        }
        result = {
            PARAMETERS: parameters,
            DEVELOPMENT: seeds,
            'development_spread': spread,
            CONFIRMATION: confirmed,
            'decisions': {
                'S1_recognition_at_scale': confirmed[RECOGNITION_PASSES],
                'S2_memory_at_scale': confirmed[MEMORY_PASSES],
                'E_every_encoder_seed_keeps_memory': all(s[MEMORY_PASSES] for s in seeds.values()),
            },
        }
        write_json(path=directory / REPORT_FILE, value=result)
    return result

"""Measured CPU throughput, sustained performance, and runtime budget estimates."""

import json
import math
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numba
import numpy as np
import psutil
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.files import brain_stats, verify_brain_files
from flystate.brain.runtime import EpisodeBrain, RestState
from flystate.experiments.config import BrainConfig, ReadoutConfig
from flystate.hashing import stable_int
from flystate.log import get_logger
from flystate.settings import Paths
from flystate.storage.json import write_json

SUSTAINED_WINDOW_SECONDS: float = 30.0
SUSTAINED_BLOCK_STEPS: int = 10
EPISODE_ESTIMATES: tuple[tuple[int, int], ...] = ((8, 3), (8, 5), (12, 5), (8, 10))


def default_threads() -> tuple[int, ...]:
    """Choose the requested benchmark grid within the logical CPU count.

    :returns: Sorted unique positive thread counts.
    :rtype: tuple[int, ...]
    """
    count = os.cpu_count() or 1
    return tuple(sorted({value for value in (1, 2, 4, 8, 12, count) if value <= count}))


def resolve_threads(paths: Paths, configured: int | None) -> int:
    """Resolve explicit threads, the latest measured optimum, or logical CPUs.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :param configured: Explicit positive thread count or automatic selection.
    :type configured: Optional[int]
    :returns: Resolved thread count, ready to enter cache identity.
    :rtype: int
    :raises ValueError: If an explicit value or saved benchmark is invalid.
    """
    if configured is not None:
        value = configured
    else:
        source = paths.runs / 'benchmarks' / 'latest.json'
        if source.exists():
            try:
                value = json.loads(s=source.read_text(encoding='utf-8'))['best']['threads']
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Invalid saved benchmark: {source}') from error
        else:
            value = os.cpu_count() or 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('Resolved thread count must be a positive integer.')
    return value


def host_info() -> dict[str, object]:
    """Describe CPU and memory resources without external commands.

    :returns: Host identity, CPU counts, decimal RAM gigabytes, and platform.
    :rtype: dict[str, object]
    """
    cpu_model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding='utf-8').splitlines():
            if line.startswith('model name'):
                cpu_model = line.partition(':')[2].strip()
                break
    return {
        'hostname': platform.node(),
        'cpu_model': cpu_model,
        'logical_cpus': psutil.cpu_count(logical=True),
        'physical_cpus': psutil.cpu_count(logical=False),
        'ram_total_gb': psutil.virtual_memory().total / 1e9,
        'platform': platform.platform(),
        'wsl': 'microsoft' in platform.release().lower(),
    }


def cost_estimates(ms_per_episode_step: float) -> dict[str, object]:
    """Compute episode seconds and single-mode cache build minutes.

    :param ms_per_episode_step: Measured milliseconds per step per fly.
    :type ms_per_episode_step: float
    :returns: Episode and trace-build estimate tables with explicit units.
    :rtype: dict[str, object]
    :raises ValueError: If the measurement is nonpositive or nonfinite.
    """
    if not math.isfinite(ms_per_episode_step) or ms_per_episode_step <= 0:
        raise ValueError('Step time must be finite and positive.')
    return {
        'episode_estimates': [
            {
                'observations': observations,
                'steps_per_observation': steps,
                'steps': observations * steps,
                'seconds_per_episode': observations * steps * ms_per_episode_step / 1000,
            }
            for observations, steps in EPISODE_ESTIMATES
        ],
        'trace_build_estimates': [
            {
                'samples': samples,
                'observations': 8,
                'steps_per_observation': steps,
                'minutes': samples * 8 * steps * ms_per_episode_step / 60000,
            }
            for samples in (400, 2000)
            for steps in (3, 5, 10)
        ],
    }


def _cpu_frequency() -> float | None:
    """Read current CPU frequency when exposed by the host.

    :returns: Frequency in MHz or null when unavailable.
    :rtype: Optional[float]
    """
    try:
        reading = psutil.cpu_freq()
        return float(reading.current) if reading is not None else None
    except (OSError, NotImplementedError):
        return None


def _window_time(
    samples: Sequence[tuple[float, float]], start: float, end: float, batch_size: int
) -> float:
    """Weight timed blocks by their overlap with a sustained observation window.

    :param samples: Consecutive block start and stop offsets in seconds.
    :type samples: Sequence[tuple[float, float]]
    :param start: Inclusive window start in seconds.
    :type start: float
    :param end: Window end in seconds.
    :type end: float
    :param batch_size: Flies simulated by each step.
    :type batch_size: int
    :returns: Window mean milliseconds per episode-step.
    :rtype: float
    """
    elapsed = 0.0
    steps = 0.0
    for left, right in samples:
        overlap = max(0.0, min(right, end) - max(left, start))
        elapsed += overlap
        steps += SUSTAINED_BLOCK_STEPS * overlap / (right - left)
    return elapsed * 1000 / (steps * batch_size)


def measure_sustained(
    runtime: EpisodeBrain,
    input_idx: NDArray[np.int64],
    currents: NDArray[np.float32],
    seconds: float,
) -> dict[str, Any]:
    """Measure first and last thirty-second windows of continuous simulation.

    :param runtime: Warmed winning runtime.
    :type runtime: EpisodeBrain
    :param input_idx: Input indices, int64 shape (I,).
    :type input_idx: NDArray[np.int64]
    :param currents: Constant float32 input voltages, shape (I,B).
    :type currents: NDArray[np.float32]
    :param seconds: Positive requested measurement duration.
    :type seconds: float
    :returns: Window timings, throttle ratio, and optional CPU frequencies.
    :rtype: dict[str, Any]
    """
    frequency_start = _cpu_frequency()
    samples: list[tuple[float, float]] = []
    origin = time.perf_counter()
    elapsed = 0.0
    next_report = SUSTAINED_WINDOW_SECONDS
    logger = get_logger(name='benchmark')
    while elapsed < seconds:
        left = time.perf_counter() - origin
        runtime.run(input_idx=input_idx, currents=currents, n_steps=SUSTAINED_BLOCK_STEPS)
        elapsed = time.perf_counter() - origin
        samples.append((left, elapsed))
        if elapsed >= next_report:
            logger.info('sustained_progress', elapsed_s=round(elapsed, 1), target_s=seconds)
            next_report += SUSTAINED_WINDOW_SECONDS
    first = _window_time(
        samples=samples,
        start=0,
        end=min(elapsed, SUSTAINED_WINDOW_SECONDS),
        batch_size=runtime.batch_size,
    )
    last = _window_time(
        samples=samples,
        start=max(0, elapsed - SUSTAINED_WINDOW_SECONDS),
        end=elapsed,
        batch_size=runtime.batch_size,
    )
    return {
        'seconds': seconds,
        'elapsed_seconds': elapsed,
        'ms_per_episode_step_first_30s': first,
        'ms_per_episode_step_last_30s': last,
        'throttle_ratio': last / first,
        'cpu_freq_mhz_start': frequency_start,
        'cpu_freq_mhz_end': _cpu_frequency(),
    }


def run_benchmark(
    paths: Paths,
    brain_dir: Path,
    threads: Sequence[int],
    batches: Sequence[int],
    warmup: int = 50,
    steps: int = 300,
    sustained_seconds: float = 300,
    input_neurons: int = 3872,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure a CPU grid, sustain its winner, and atomically publish a report.

    :param paths: Output storage home.
    :type paths: Paths
    :param brain_dir: Explicit installed brain directory.
    :type brain_dir: Path
    :param threads: Positive Numba thread grid.
    :type threads: Sequence[int]
    :param batches: Positive batch-size grid.
    :type batches: Sequence[int]
    :param warmup: Untimed input-driven steps before each measurement.
    :type warmup: int
    :param steps: Timed steps per grid cell.
    :type steps: int
    :param sustained_seconds: Positive duration of sustained measurement.
    :type sustained_seconds: float
    :param input_neurons: Distinct visual projection neurons to stimulate.
    :type input_neurons: int
    :param seed: Seed for input selection and episode noise.
    :type seed: int
    :returns: Complete persisted benchmark report, including its output path.
    :rtype: dict[str, Any]
    :raises ValueError: If grid values, duration, or input population size are invalid.
    """
    if not threads or not batches or min(*threads, *batches, steps, input_neurons) < 1:
        raise ValueError('Thread, batch, step, and input counts must be positive.')
    if warmup < 0 or seed < 0 or not math.isfinite(sustained_seconds) or sustained_seconds <= 0:
        raise ValueError(
            'Warmup and seed must be nonnegative; sustained duration must be positive.'
        )
    brain_cfg = BrainConfig()
    readout_cfg = ReadoutConfig()
    logger = get_logger(name='benchmark')
    process = psutil.Process()
    original_threads = numba.get_num_threads()
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            started = time.perf_counter()
            runtime = EpisodeBrain(
                brain_dir=brain_dir,
                brain_cfg=brain_cfg,
                readout_cfg=readout_cfg,
                batch_size=1,
                threads=threads[0],
            )
            load = {
                'load_s': time.perf_counter() - started,
                'rss_mb_after_load': process.memory_info().rss / 1e6,
            }
            population = runtime.cells(superclasses=['visual_projection'])
            if input_neurons > len(population):
                raise ValueError(
                    f'Requested {input_neurons} inputs; population has {len(population)}.'
                )
            generator = np.random.default_rng(
                seed=np.random.SeedSequence(entropy=[seed, stable_int(key='benchmark-input')])
            )
            indices = np.sort(a=generator.choice(a=population, size=input_neurons, replace=False))
            drive = generator.uniform(low=-0.1, high=0.1, size=(input_neurons, 1)).astype(
                np.float32
            )
            zero = RestState(
                v=np.zeros(shape=runtime.n, dtype=np.float32),
                fired=np.empty(shape=0, dtype=np.int64),
                warmup_steps=0,
                seed=seed,
            )
            runtime.begin(rest=zero, sample_ids=['compile'], seed=seed)
            started = time.perf_counter()
            runtime.run(input_idx=indices, currents=drive, n_steps=1)
            load['first_step_s'] = time.perf_counter() - started
            del runtime
            grid: list[dict[str, Any]] = []
            for thread_count in dict.fromkeys(threads):
                for batch in dict.fromkeys(batches):
                    runtime = EpisodeBrain(
                        brain_dir=brain_dir,
                        brain_cfg=brain_cfg,
                        readout_cfg=readout_cfg,
                        batch_size=batch,
                        threads=thread_count,
                    )
                    rest = runtime.compute_rest_state(seed=seed)
                    runtime.begin(
                        rest=rest, sample_ids=[f'bench-{i}' for i in range(batch)], seed=seed
                    )
                    currents = np.repeat(a=drive, repeats=batch, axis=1)
                    runtime.run(input_idx=indices, currents=currents, n_steps=warmup)
                    psutil.cpu_percent(interval=None)
                    started = time.perf_counter()
                    summary = runtime.run(input_idx=indices, currents=currents, n_steps=steps)
                    elapsed = time.perf_counter() - started
                    row = {
                        'threads': thread_count,
                        'batch': batch,
                        'steps': steps,
                        'ms_per_step': elapsed * 1000 / steps,
                        'ms_per_episode_step': elapsed * 1000 / (steps * batch),
                        'steps_per_s': steps / elapsed,
                        'cpu_percent': psutil.cpu_percent(interval=None),
                        'mean_rate_hz': float(summary.spikes_total.sum())
                        / (runtime.n * steps * batch * brain_cfg.dt_s),
                        'rss_mb': process.memory_info().rss / 1e6,
                    }
                    grid.append(row)
                    logger.info('grid_measured', **row)
                    del runtime
            best_row = min(grid, key=lambda row: row['ms_per_episode_step'])
            best = {key: best_row[key] for key in ('threads', 'batch', 'ms_per_episode_step')}
            runtime = EpisodeBrain(
                brain_dir=brain_dir,
                brain_cfg=brain_cfg,
                readout_cfg=readout_cfg,
                batch_size=best['batch'],
                threads=best['threads'],
            )
            runtime.begin(
                rest=runtime.compute_rest_state(seed=seed),
                sample_ids=[f'bench-{i}' for i in range(best['batch'])],
                seed=seed,
            )
            currents = np.repeat(a=drive, repeats=best['batch'], axis=1)
            runtime.run(input_idx=indices, currents=currents, n_steps=warmup)
            sustained = measure_sustained(
                runtime=runtime, input_idx=indices, currents=currents, seconds=sustained_seconds
            )
    finally:
        numba.set_num_threads(n=original_threads)
    now = datetime.now(tz=UTC)
    directory = paths.runs / 'benchmarks'
    stem = f'{now:%Y%m%d-%H%M%S}-{platform.node()}'
    destination = directory / f'{stem}.json'
    suffix = 1
    while destination.exists():
        destination = directory / f'{stem}-{suffix}.json'
        suffix += 1
    report = {
        'schema_version': 1,
        'created_utc': now.isoformat(),
        'host': host_info(),
        'versions': {
            **{
                name: version(distribution_name=name)
                for name in ('flystate', 'flybrain', 'numpy', 'numba')
            },
            'python': platform.python_version(),
        },
        'brain': {
            **brain_stats(brain_dir=brain_dir),
            'brain_dir': str(brain_dir),
            'files': verify_brain_files(brain_dir=brain_dir),
        },
        'parameters': {'seed': seed, 'input_neurons': input_neurons, 'warmup': warmup},
        'load': load,
        'grid': grid,
        'best': best,
        'sustained': sustained,
        **cost_estimates(ms_per_episode_step=sustained['ms_per_episode_step_last_30s']),
        'report_path': str(destination),
    }
    write_json(path=destination, value=report)
    write_json(path=directory / 'latest.json', value=report)
    return report

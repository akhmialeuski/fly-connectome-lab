"""Matched-noise measurements of input firing regime and readout response latency."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numba
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.benchmark import resolve_threads
from flystate.brain.runtime import EpisodeBrain, RestState
from flystate.datasets.preprocess import prepare_dataset
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import Episode, EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.log import get_logger
from flystate.settings import Paths
from flystate.storage.json import write_json

DEFAULT_AMPLITUDES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
CANDIDATE_STEPS: tuple[int, ...] = (3, 5, 10)
RESPONSE_THRESHOLD: float = 1e-6
MAX_RATE_HZ: float = 5.0
MIN_READOUT_RATE_HZ: float = 0.05


def response_latencies(differences: NDArray) -> tuple[int | None, int | None]:
    """Measure first detectable and half-observed-peak response times in steps.

    :param differences: Nonnegative finite median voltage differences, shape (T,), T>0.
    :type differences: NDArray
    :returns: One-based threshold and half-peak crossings, or absent responses.
    :rtype: tuple[Optional[int], Optional[int]]
    :raises ValueError: If the response curve is empty, negative, or nonfinite.
    """
    if (
        differences.ndim != 1
        or not len(differences)
        or not np.isfinite(differences).all()
        or np.any(differences < 0)
    ):
        raise ValueError('Response curve must be a nonempty finite nonnegative vector.')
    detected = np.flatnonzero(a=differences > RESPONSE_THRESHOLD)
    peak = float(differences.max())
    half = np.flatnonzero(a=differences >= 0.5 * peak) if peak > 0 else np.empty(shape=0, dtype=int)
    return (
        int(detected[0]) + 1 if len(detected) else None,
        int(half[0]) + 1 if len(half) else None,
    )


def recommend(
    amplitudes: Sequence[dict[str, Any]],
) -> tuple[dict[str, float | int | None], list[str]]:
    """Select the smallest measured admissible amplitude and latency-compatible duration.

    :param amplitudes: Measured amplitude rows with latency and readout rate fields.
    :type amplitudes: Sequence[dict[str, Any]]
    :returns: Nullable recommendation and explicit missing-admissibility warnings.
    :rtype: tuple[dict[str, float | int | None], list[str]]
    """
    valid = [
        row
        for row in amplitudes
        if row['latency_first'] is not None
        and row['latency_half'] is not None
        and row['rate_readout_hz'] <= MAX_RATE_HZ
    ]
    if not valid:
        return {'amplitude': None, 'steps_per_observation': None}, ['no_admissible_amplitude']
    selected = min(valid, key=lambda row: row['amplitude'])
    step = next((value for value in CANDIDATE_STEPS if value >= selected['latency_half']), None)
    if step is None:
        return {'amplitude': None, 'steps_per_observation': None}, ['latency_exceeds_10']
    return {'amplitude': selected['amplitude'], 'steps_per_observation': step}, []


def _spontaneous(
    brain: EpisodeBrain, rest: RestState, ids: list[str], seed: int, steps: int, dt: float
) -> dict[str, float]:
    """Measure rates over real probes, excluding unused padded flies.

    :param brain: Runtime reused by all measurements.
    :type brain: EpisodeBrain
    :param rest: Common warmed single-fly state.
    :type rest: RestState
    :param ids: Independent probe identifiers.
    :type ids: list[str]
    :param seed: Episode noise seed.
    :type seed: int
    :param steps: Spontaneous measurement duration in steps.
    :type steps: int
    :param dt: Simulation seconds per step.
    :type dt: float
    :returns: Population-average all-neuron and readout rates in hertz.
    :rtype: dict[str, float]
    """
    total = 0
    readout = 0
    for start in range(0, len(ids), brain.batch_size):
        batch = ids[start : start + brain.batch_size]
        brain.begin(rest=rest, sample_ids=batch, seed=seed)
        summary = brain.run(
            input_idx=np.empty(shape=0, dtype=np.int64), currents=None, n_steps=steps
        )
        total += int(summary.spikes_total[: len(batch)].sum())
        readout += int(summary.readout_spikes[: len(batch)].sum())
    duration = len(ids) * steps * dt
    return {
        'rate_all_hz': total / (brain.n * duration),
        'rate_readout_hz': readout / (len(brain.readout_idx) * duration),
    }


def _probe_amplitude(
    brain: EpisodeBrain,
    rest: RestState,
    episodes: list[Episode],
    cfg: ExperimentConfig,
    amplitude: float,
    steps: int,
) -> dict[str, Any]:
    """Stimulate paired flies with shared noise and measure their readout divergence.

    :param brain: Even-sized runtime with at least one matched pair.
    :type brain: EpisodeBrain
    :param rest: Common warmed state for every pair.
    :type rest: RestState
    :param episodes: Validation probe episodes in stable sample order.
    :type episodes: list[Episode]
    :param cfg: Validated experiment and noise settings.
    :type cfg: ExperimentConfig
    :param amplitude: Validated positive voltage scale.
    :type amplitude: float
    :param steps: Measurement duration in simulation steps.
    :type steps: int
    :returns: Firing rates, median voltage curve, latencies, and encoder provenance.
    :rtype: dict[str, Any]
    """
    encoder = SparseProjectionEncoder(
        cfg=cfg.encoder.model_copy(update={'amplitude': amplitude}),
        window=cfg.episodes.window,
        candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
    )
    differences = np.empty(shape=(len(episodes), steps), dtype=np.float64)
    counts = np.zeros(shape=3, dtype=np.int64)
    pairs = brain.batch_size // 2
    for start in range(0, len(episodes), pairs):
        batch = episodes[start : start + pairs]
        ids = [episode.sample_id for episode in batch for _ in range(2)]
        brain.begin(rest=rest, sample_ids=ids, seed=cfg.seed)
        encoded = encoder.encode(
            observations=np.stack(arrays=[e.observations[0] for e in batch]),
            positions=np.stack(arrays=[e.positions[0] for e in batch]),
        )
        currents = np.zeros(shape=(len(encoder.input_idx), brain.batch_size), dtype=np.float32)
        currents[:, : 2 * len(batch) : 2] = encoded
        for step in range(steps):
            summary = brain.run(input_idx=encoder.input_idx, currents=currents, n_steps=1)
            counts += [
                summary.spikes_total[: 2 * len(batch) : 2].sum(),
                summary.input_spikes[: 2 * len(batch) : 2].sum(),
                summary.readout_spikes[: 2 * len(batch) : 2].sum(),
            ]
            voltage = brain.features(kinds=['voltage'])[: 2 * len(batch)]
            differences[start : start + len(batch), step] = np.abs(
                voltage[::2] - voltage[1::2]
            ).mean(axis=1)
    curve = np.median(a=differences, axis=0)
    first, half = response_latencies(differences=curve)
    duration = len(episodes) * steps * cfg.brain.dt_s
    return {
        'amplitude': amplitude,
        'rate_all_hz': float(counts[0] / (brain.n * duration)),
        'rate_input_hz': float(counts[1] / (len(encoder.input_idx) * duration)),
        'rate_readout_hz': float(counts[2] / (len(brain.readout_idx) * duration)),
        'd': curve.tolist(),
        'latency_first': first,
        'latency_half': half,
        'encoder': encoder.spec(),
    }


def calibrate(
    cfg: ExperimentConfig,
    paths: Paths,
    brain_dir: Path,
    amplitudes: Sequence[float] = DEFAULT_AMPLITUDES,
    probes: int = 32,
    steps: int = 10,
    spontaneous_steps: int = 50,
) -> dict[str, Any]:
    """Persist matched-noise validation calibration without editing experiment settings.

    :param cfg: Validated experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage home.
    :type paths: Paths
    :param brain_dir: Explicit installed brain directory.
    :type brain_dir: Path
    :param amplitudes: Nonempty finite amplitude grid in (0,1].
    :type amplitudes: Sequence[float]
    :param probes: Positive number of validation images.
    :type probes: int
    :param steps: Positive driven/control measurement steps.
    :type steps: int
    :param spontaneous_steps: Positive unstimulated measurement steps.
    :type spontaneous_steps: int
    :returns: Persisted calibration report with nullable recommendations.
    :rtype: dict[str, Any]
    :raises ValueError: If settings are invalid or validation images are insufficient.
    """
    if (
        min(probes, steps, spontaneous_steps) < 1
        or not amplitudes
        or any(not np.isfinite(value) or not 0 < value <= 1 for value in amplitudes)
    ):
        raise ValueError('Calibration requires positive counts and finite amplitudes in (0,1].')
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    validation = [(i, sample) for i, sample in enumerate(prepared.samples) if sample.split == 'val']
    if probes > len(validation):
        raise ValueError(
            f'Requested {probes} probes but only {len(validation)} validation images exist.'
        )
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    episodes = [
        builder.build(sample=sample, image=prepared.images[i]) for i, sample in validation[:probes]
    ]
    threads = resolve_threads(paths=paths, configured=cfg.brain.threads)
    original_threads = numba.get_num_threads()
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            brain = EpisodeBrain(
                brain_dir=brain_dir,
                brain_cfg=cfg.brain,
                readout_cfg=cfg.readout,
                batch_size=2 * max(1, cfg.brain.batch_size // 2),
                threads=threads,
            )
            rest = brain.compute_rest_state(seed=cfg.seed)
            spontaneous = _spontaneous(
                brain=brain,
                rest=rest,
                ids=[e.sample_id for e in episodes],
                seed=cfg.seed,
                steps=spontaneous_steps,
                dt=cfg.brain.dt_s,
            )
            results = []
            for amplitude in sorted(set(amplitudes)):
                get_logger(name='calibration').info(
                    'probe_amplitude', amplitude=amplitude, probes=probes
                )
                results.append(
                    _probe_amplitude(
                        brain=brain,
                        rest=rest,
                        episodes=episodes,
                        cfg=cfg,
                        amplitude=amplitude,
                        steps=steps,
                    )
                )
    finally:
        numba.set_num_threads(n=original_threads)
    recommendation, warnings = recommend(amplitudes=results)
    if spontaneous['rate_all_hz'] > MAX_RATE_HZ:
        warnings.append('runaway')
    if spontaneous['rate_readout_hz'] < MIN_READOUT_RATE_HZ:
        warnings.append('silent_readout')
    warnings.extend(
        f'no_response:{row["amplitude"]}' for row in results if row['latency_first'] is None
    )
    now = datetime.now(tz=UTC)
    destination = (
        paths.runs / 'calibrations' / f'{now:%Y%m%d-%H%M%S}-{cfg.name}-{uuid4().hex[:8]}.json'
    )
    report = {
        'schema_version': 1,
        'created_utc': now.isoformat(),
        'config_name': cfg.name,
        'config_hash': config_hash(cfg=cfg),
        'threads': threads,
        'paired_batch_size': brain.batch_size,
        'dataset_fingerprint': prepared.fingerprint,
        'probe_ids': [e.sample_id for e in episodes],
        'steps': steps,
        'spontaneous_steps': spontaneous_steps,
        'brain_files_sha256': {
            name: sha256_file(path=brain_dir / name) for name in ('brain.npz', 'weights.npz')
        },
        'spontaneous': spontaneous,
        'amplitudes': results,
        'warnings': warnings,
        'recommendation': recommendation,
        'output': str(destination),
    }
    write_json(path=destination, value=report)
    return report

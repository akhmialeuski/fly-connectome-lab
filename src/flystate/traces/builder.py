"""Bounded-memory simulation with single-writer exclusion and resumable chunk commits."""

import shutil
import time
from collections import Counter, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
import numpy as np
import psutil
from filelock import FileLock
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.benchmark import resolve_threads
from flystate.brain.runtime import EpisodeBrain, RestState
from flystate.datasets.preprocess import PreparedDataset, prepare_dataset
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file
from flystate.log import configure_logging, get_logger
from flystate.settings import Paths
from flystate.traces.key import cache_key, cache_key_fields, simulation_mode
from flystate.traces.store import TraceError, TraceStore, directory_size, index_rows

PROGRESS_SECONDS: float = 10.0
RATE_WINDOW_SECONDS: float = 300.0


def resolve_trace(
    cfg: ExperimentConfig, paths: Paths
) -> tuple[PreparedDataset, dict[str, Any], Path]:
    """Resolve actual numerical identity without constructing a simulator.

    :param cfg: Validated experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage home.
    :type paths: Paths
    :returns: Prepared dataset, identity fields, and cache directory.
    :rtype: tuple[PreparedDataset, dict[str, Any], Path]
    """
    threads = resolve_threads(paths=paths, configured=cfg.brain.threads)
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    digests = {name: sha256_file(path=paths.brain / name) for name in ('brain.npz', 'weights.npz')}
    fields = cache_key_fields(cfg=cfg, prepared=prepared, brain_files_sha=digests, threads=threads)
    return prepared, fields, paths.features / cache_key(fields=fields)


def open_trace(cfg: ExperimentConfig, paths: Paths, require_complete: bool = True) -> TraceStore:
    """Resolve and verify a cache for inspection or downstream readout fitting.

    :param cfg: Validated experiment settings.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage home.
    :type paths: Paths
    :param require_complete: Require a fully completed build.
    :type require_complete: bool
    :returns: Read-only verified trace store.
    :rtype: TraceStore
    :raises TraceError: If the expected cache is absent, incompatible, or corrupt.
    """
    prepared, fields, directory = resolve_trace(cfg=cfg, paths=paths)
    store = TraceStore(directory=directory)
    if store.meta['key_fields'] != fields or store.index != index_rows(samples=prepared.samples):
        raise TraceError('Trace identity or index does not match the prepared dataset.')
    store.verify(require_complete=require_complete)
    return store


def _simulate_chunk(
    cfg: ExperimentConfig,
    prepared: PreparedDataset,
    builder: EpisodeBuilder,
    brain: EpisodeBrain,
    encoder: SparseProjectionEncoder,
    rest: RestState,
    start: int,
    stop: int,
    progress: Callable[[int], None],
) -> tuple[NDArray, NDArray, NDArray]:
    """Simulate one bounded chunk, excluding padded flies from stored features and counts.

    :param cfg: Experiment and state-reset settings.
    :type cfg: ExperimentConfig
    :param prepared: Canonical aligned images and sample records.
    :type prepared: PreparedDataset
    :param builder: Shared observation geometry builder.
    :type builder: EpisodeBuilder
    :param brain: Fixed-thread batched simulation runtime.
    :type brain: EpisodeBrain
    :param encoder: Frozen input-current mapping.
    :type encoder: SparseProjectionEncoder
    :param rest: Shared warmed state.
    :type rest: RestState
    :param start: First included dataset row.
    :type start: int
    :param stop: Exclusive final dataset row.
    :type stop: int
    :param progress: Callback receiving each completed real batch size.
    :type progress: Callable[[int], None]
    :returns: Float32 features (rows,T,F), counts (rows,T,3), and simulation milliseconds (rows,).
    :rtype: tuple[NDArray, NDArray, NDArray]
    """
    features = np.empty(
        shape=(
            stop - start,
            cfg.episodes.steps,
            len(brain.readout_idx) * len(cfg.readout.features),
        ),
        dtype=np.float32,
    )
    summaries = np.empty(shape=(stop - start, cfg.episodes.steps, 3), dtype=np.float32)
    timings = np.empty(shape=stop - start, dtype=np.float32)
    reset = simulation_mode(cfg=cfg) == 'reset'
    for first in range(start, stop, brain.batch_size):
        end = min(first + brain.batch_size, stop)
        episodes = [
            builder.build(sample=prepared.samples[row], image=prepared.images[row])
            for row in range(first, end)
        ]
        before = time.perf_counter()
        brain.begin(
            rest=rest, sample_ids=[episode.sample_id for episode in episodes], seed=cfg.seed
        )
        seconds = time.perf_counter() - before
        count = end - first
        for step in range(cfg.episodes.steps):
            currents = np.zeros(shape=(len(encoder.input_idx), brain.batch_size), dtype=np.float32)
            currents[:, :count] = encoder.encode(
                observations=np.stack(arrays=[episode.observations[step] for episode in episodes]),
                positions=np.stack(arrays=[episode.positions[step] for episode in episodes]),
            )
            before = time.perf_counter()
            if reset:
                brain.restore_rest(rest=rest)
            summary = brain.run(
                input_idx=encoder.input_idx,
                currents=currents,
                n_steps=cfg.brain.steps_per_observation,
            )
            values = brain.features(kinds=cfg.readout.features)
            seconds += time.perf_counter() - before
            features[first - start : end - start, step] = values[:count]
            summaries[first - start : end - start, step] = np.stack(
                arrays=[
                    summary.spikes_total[:count],
                    summary.active_neurons[:count],
                    summary.readout_spikes[:count],
                ],
                axis=1,
            )
        timings[first - start : end - start] = seconds * 1000 / count
        progress(count)
    return features, summaries, timings


def _progress(
    store: TraceStore,
    history: deque[tuple[float, int]],
    started: float,
    processed: int,
    cfg: ExperimentConfig,
) -> None:
    """Log bounded-window throughput, remaining work, memory, and storage use.

    :param store: Current cache handle.
    :type store: TraceStore
    :param history: Recent monotonic completion times and cumulative attempt row counts.
    :type history: deque[tuple[float, int]]
    :param started: Monotonic attempt start time.
    :type started: float
    :param processed: Total completed rows including previous attempts.
    :type processed: int
    :param cfg: Simulation step counts.
    :type cfg: ExperimentConfig
    """
    now, current = history[-1]
    while len(history) > 2 and history[1][0] < now - RATE_WINDOW_SECONDS:
        history.popleft()
    elapsed = max(now - history[0][0], 1e-9)
    rate = (current - history[0][1]) / elapsed
    get_logger(name='trace').info(
        'trace_progress',
        processed=processed,
        total=store.n,
        samples_per_second=rate,
        brain_steps_per_second=rate * cfg.episodes.steps * cfg.brain.steps_per_observation,
        elapsed_seconds=now - started,
        eta_seconds=(store.n - processed) / rate if rate else None,
        rss_gb=psutil.Process().memory_info().rss / 1e9,
        cache_mb=directory_size(directory=store.directory) / 1e6,
        cache_key=store.meta['key'],
    )


def build_trace(
    cfg: ExperimentConfig,
    paths: Paths,
    force: bool = False,
    max_chunks: int | None = None,
    verbosity: int = 0,
) -> dict[str, Any]:
    """Build or resume exactly one writer's trace cache and preserve committed chunks.

    :param cfg: Validated experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage home.
    :type paths: Paths
    :param force: Remove only this resolved cache while holding its writer lock.
    :type force: bool
    :param max_chunks: Optional positive chunk budget for a resumable trial build.
    :type max_chunks: Optional[int]
    :param verbosity: Console logging verbosity preserved when binding cache logs.
    :type verbosity: int
    :returns: Cache identity, location, status, dimensions, and completion counts.
    :rtype: dict[str, Any]
    :raises TraceError: If settings, stored data, or simulation output are invalid.
    :raises KeyboardInterrupt: After persisting interrupted status on user interruption.
    """
    if max_chunks is not None and max_chunks < 1:
        raise TraceError('max_chunks must be positive.')
    prepared, fields, directory = resolve_trace(cfg=cfg, paths=paths)
    locks = paths.features / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_file=locks / f'{directory.name}.lock', timeout=0):
        if force and directory.exists():
            shutil.rmtree(path=directory)
        store = TraceStore(directory=directory, mode='a') if directory.exists() else None
        if store is not None:
            if store.meta['key_fields'] != fields or store.index != index_rows(
                samples=prepared.samples
            ):
                raise TraceError('Stored trace identity or index differs from requested samples.')
            store.verify()
            if store.meta['status'] == 'completed':
                return inspect_trace(store=store)
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
                encoder = SparseProjectionEncoder(
                    cfg=cfg.encoder,
                    window=cfg.episodes.window,
                    candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
                )
                if store is None:
                    store = TraceStore.create(
                        directory=directory,
                        fields=fields,
                        samples=prepared.samples,
                        steps=cfg.episodes.steps,
                        features=len(brain.readout_idx) * len(cfg.readout.features),
                        threads=brain.threads,
                        batch_size=brain.batch_size,
                    )
                store.set_status(
                    status='running', batch_size=brain.batch_size, encoder=encoder.spec()
                )
                configure_logging(verbosity=verbosity, json_path=directory / 'events.jsonl')
                _build_chunks(
                    store=store,
                    cfg=cfg,
                    prepared=prepared,
                    brain=brain,
                    encoder=encoder,
                    max_chunks=max_chunks,
                )
        except KeyboardInterrupt:
            if store is not None and store.meta['status'] != 'completed':
                store.set_status(status='interrupted')
            raise
        except Exception as error:
            if store is not None and store.meta['status'] != 'completed':
                store.set_status(status='failed', error=str(error))
            raise
        finally:
            numba.set_num_threads(n=original_threads)
            configure_logging(verbosity=verbosity)
        return inspect_trace(store=store)


def _build_chunks(
    store: TraceStore,
    cfg: ExperimentConfig,
    prepared: PreparedDataset,
    brain: EpisodeBrain,
    encoder: SparseProjectionEncoder,
    max_chunks: int | None,
) -> None:
    """Commit remaining chunks and persist the final status for this attempt.

    :param store: Locked writable cache.
    :type store: TraceStore
    :param cfg: Validated simulation settings.
    :type cfg: ExperimentConfig
    :param prepared: Verified aligned dataset.
    :type prepared: PreparedDataset
    :param brain: Fixed-thread simulation runtime.
    :type brain: EpisodeBrain
    :param encoder: Frozen neuron/current mapping.
    :type encoder: SparseProjectionEncoder
    :param max_chunks: Optional attempt chunk budget.
    :type max_chunks: Optional[int]
    """
    done = store.verify()
    rest = brain.compute_rest_state(seed=cfg.seed)
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    started = time.perf_counter()
    history = deque([(started, 0)])
    attempted = 0
    processed = sum(
        store.bounds(chunk=int(c))[1] - store.bounds(chunk=int(c))[0]
        for c in np.flatnonzero(a=done)
    )
    last_progress = started

    def on_batch(count: int) -> None:
        """Record simulated rows and emit progress between batches.

        :param count: Newly simulated real samples.
        :type count: int
        """
        nonlocal attempted, processed, last_progress
        attempted += count
        processed += count
        now = time.perf_counter()
        history.append((now, attempted))
        if now - last_progress >= PROGRESS_SECONDS:
            _progress(store=store, history=history, started=started, processed=processed, cfg=cfg)
            last_progress = now

    for written, chunk in enumerate(np.flatnonzero(a=~done), start=1):
        start, stop = store.bounds(chunk=int(chunk))
        features, summary, timings = _simulate_chunk(
            cfg=cfg,
            prepared=prepared,
            builder=builder,
            brain=brain,
            encoder=encoder,
            rest=rest,
            start=start,
            stop=stop,
            progress=on_batch,
        )
        store.write_chunk(chunk=int(chunk), features=features, summary=summary, brain_ms=timings)
        get_logger(name='trace').info(
            'chunk_written', cache_key=store.meta['key'], chunk=int(chunk), rows=stop - start
        )
        if max_chunks is not None and written >= max_chunks:
            break
    _progress(store=store, history=history, started=started, processed=processed, cfg=cfg)
    store.set_status(
        status='completed' if store.verify().all() else 'interrupted',
        attempt_elapsed_seconds=time.perf_counter() - started,
    )


def inspect_trace(store: TraceStore) -> dict[str, Any]:
    """Summarize a verified cache without loading the full feature tensor.

    :param store: Open trace cache.
    :type store: TraceStore
    :returns: Machine-readable cache dimensions and build state.
    :rtype: dict[str, Any]
    """
    done = store.verify()
    return {
        **{
            key: store.meta[key]
            for key in ('key', 'status', 'N', 'T', 'F', 'threads', 'batch_size')
        },
        'directory': str(store.directory),
        'chunks_done': int(done.sum()),
        'chunks_total': store.chunks,
        'size_bytes': directory_size(directory=store.directory),
    }


def trace_stats(store: TraceStore) -> dict[str, Any]:
    """Compute per-observation moments from committed rows in bounded chunk reads.

    :param store: Completed and verified trace cache.
    :type store: TraceStore
    :returns: Observation means, zero-trace fractions, and split counts.
    :rtype: dict[str, Any]
    """
    store.verify(require_complete=True)
    kinds = store.meta['key_fields']['readout']['features']
    population = store.features_count // len(kinds)
    totals = np.zeros(shape=(store.steps, 6), dtype=np.float64)
    for chunk in range(store.chunks):
        start, stop = store.bounds(chunk=chunk)
        features = np.asarray(a=store.array(name='features')[start:stop], dtype=np.float64)
        totals[:, 3:] += np.asarray(
            a=store.array(name='summary')[start:stop], dtype=np.float64
        ).sum(axis=0)
        for index, kind in enumerate(kinds):
            values = features[:, :, index * population : (index + 1) * population]
            if kind == 'spike_trace':
                totals[:, 0] += values.sum(axis=(0, 2)) / population
                totals[:, 1] += (values == 0).sum(axis=(0, 2)) / population
            else:
                totals[:, 2] += values.sum(axis=(0, 2)) / population
    totals /= store.n
    names = (
        'mean_spike_trace',
        'zero_spike_trace_fraction',
        'mean_voltage',
        'mean_spikes_total',
        'mean_active_neurons',
        'mean_readout_spikes',
    )
    rows = []
    for step in range(store.steps):
        row = dict(zip(names, totals[step].tolist(), strict=True))
        if 'spike_trace' not in kinds:
            row['mean_spike_trace'] = row['zero_spike_trace_fraction'] = None
        if 'voltage' not in kinds:
            row['mean_voltage'] = None
        rows.append({'t': step + 1, **row})
    return {
        'key': store.meta['key'],
        'observations': rows,
        'counts_per_split': dict(Counter(row['split'] for row in store.index)),
    }

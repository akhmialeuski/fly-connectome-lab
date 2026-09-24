"""Preregistered stimulus, blank, and recovery measurements in a fixed fly graph."""

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numba
import numpy as np
import psutil
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.brain.runtime import EpisodeBrain, RestState, TemporalResponse
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.preprocess import PreparedDataset, prepare_dataset
from flystate.datasets.registry import require_validated
from flystate.diagnostics.artifacts import attempt
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, sha256_obj, stable_int
from flystate.settings import Paths
from flystate.storage.json import write_json

RESOURCE_LIMIT_BYTES: int = 12 * 1024**3
CASE_TIME_LIMIT_SECONDS: int = 3600


@dataclass(frozen=True)
class TemporalCase:
    """One fixed stimulus duration, current scale, and episode-noise treatment."""

    name: str
    stimulus_steps: int
    current_scale: float
    recovery_steps: int
    noise_enabled: bool
    hypothesis: str


CASES: dict[str, TemporalCase] = {
    'C0': TemporalCase('C0', 20, 0.0, 10, True, 'Measure the time-matched noisy blank state.'),
    'C1': TemporalCase('C1', 10, 1.0, 10, True, 'Measure the original ten-step response.'),
    'C2': TemporalCase('C2', 20, 1.0, 10, True, 'Double duration at a fixed voltage kick.'),
    'C3': TemporalCase('C3', 20, 0.5, 10, True, 'Double duration at fixed total drive.'),
    'C4': TemporalCase('C4', 10, 2.0, 10, True, 'Double the voltage kick at fixed duration.'),
    'C5': TemporalCase(
        'C5', 10, 1.0, 10, False, 'Measure the original drive without episode noise.'
    ),
    'C6': TemporalCase('C6', 20, 0.0, 10, False, 'Measure the time-matched noise-off blank.'),
}


def _read_json(path: Path) -> dict[str, Any]:
    """Read a finite protocol object without accepting arrays or pickle.

    :param path: Existing JSON file.
    :type path: Path
    :returns: Parsed object.
    :rtype: dict[str, Any]
    :raises ValueError: If the document is not an object.
    """
    value = json.loads(s=path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Protocol document must be a JSON object.')
    return value


def _verify_cohort(
    cfg: ExperimentConfig,
    paths: Paths,
    prepared: PreparedDataset,
    cohort: dict[str, Any],
    membership: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    """Check frozen train-only members and every source/aligned image hash.

    :param cfg: Dataset and split specification.
    :type cfg: ExperimentConfig
    :param paths: Registered dataset boundary.
    :type paths: Paths
    :param prepared: Verified aligned images (N,S,S,3), uint8.
    :type prepared: PreparedDataset
    :param cohort: Frozen selected IDs, labels, and image hashes.
    :type cohort: dict[str, Any]
    :param membership: Parent train/validation member lists and canonical digest.
    :type membership: dict[str, Any]
    :returns: Ordered cache row and frozen member pairs.
    :rtype: list[tuple[int, dict[str, Any]]]
    :raises ValueError: If source identity, selection, or image bytes differ.
    """
    members = membership['sample_ids_by_split']
    digest = sha256_obj(obj=members)
    if (
        digest != membership['membership_sha256']
        or digest != cohort['source_membership_sha256']
        or config_hash(cfg=cfg) != cohort['source_config_sha256']
        or prepared.key != cohort['preprocess_cache_key']
    ):
        raise ValueError('Cohort membership or preprocessing identity differs from the protocol.')
    rows = {sample.sample_id: (index, sample) for index, sample in enumerate(prepared.samples)}
    frozen = cohort['samples']
    selected: list[tuple[int, dict[str, Any]]] = []
    if len({item['sample_id'] for item in frozen}) != len(frozen):
        raise ValueError('Frozen cohort contains repeated sample IDs.')
    image_root = (
        CelebAAdapter(root=require_validated(paths=paths, name=cfg.dataset.name), expected=None)
        .locate()
        .images_dir
    )
    labels = {item['label'] for item in frozen}
    for label in labels:
        count = sum(item['label'] == label for item in frozen)
        candidates = sorted(
            (
                sample.sample_id
                for sample in prepared.samples
                if sample.split == 'train' and sample.label == label
            ),
            key=lambda sample_id: (stable_int(key=sample_id), sample_id),
        )
        expected = set(candidates[:count])
        actual = {item['sample_id'] for item in frozen if item['label'] == label}
        if actual != expected:
            raise ValueError('Frozen cohort differs from the declared rank selection rule.')
    for item in frozen:
        sample_id = item['sample_id']
        if sample_id not in members['train'] or sample_id not in rows:
            raise ValueError('Pilot image is absent from the parent training cohort.')
        index, sample = rows[sample_id]
        aligned_digest = hashlib.sha256(string=prepared.images[index].tobytes()).hexdigest()
        if (
            sample.split != 'train'
            or sample.label != item['label']
            or sha256_file(path=image_root / sample.filename) != item['source_sha256']
            or aligned_digest != item['aligned_sha256']
        ):
            raise ValueError(f'Frozen image content or label differs: {sample_id}.')
        selected.append((index, item))
    windows = cohort['windows']
    if windows != sorted(set(windows)) or any(
        window < 0 or window >= cfg.episodes.steps for window in windows
    ):
        raise ValueError('Frozen observation windows are invalid.')
    return selected


def _verify_masks(paths: Paths, document: dict[str, Any]) -> dict[str, NDArray[np.int64]]:
    """Verify selected bodyIds, indices, graph source hashes, and disjointness.

    :param paths: Brain storage boundary.
    :type paths: Paths
    :param document: Frozen population audit with sorted selected neuron indices.
    :type document: dict[str, Any]
    :returns: Named int64 index arrays, each shape (K,).
    :rtype: dict[str, NDArray[np.int64]]
    :raises ValueError: If mask provenance or anatomy differs.
    """
    if any(
        sha256_file(path=paths.brain / filename) != document[f'{key}_sha256']
        for filename, key in (('brain.npz', 'brain'), ('weights.npz', 'weights'))
    ):
        raise ValueError('Brain source files differ from frozen mask provenance.')
    populations: dict[str, NDArray[np.int64]] = {}
    occupied: set[int] = set()
    with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as meta:
        ids = meta['ids']
        superclass = meta['superclass'].astype(str)
    for name, item in document['masks'].items():
        indices = np.asarray(a=item['indices'], dtype=np.int64)
        body_ids = ids[indices].astype(np.int64).tolist()
        candidates = np.flatnonzero(a=superclass == name)
        ranked = sorted(
            candidates,
            key=lambda index: (stable_int(key=str(ids[index])), int(ids[index])),
        )
        expected = (
            candidates
            if len(candidates) == len(indices)
            else np.sort(a=np.asarray(a=ranked[: len(indices)], dtype=np.int64))
        )
        if (
            len(indices) != item['selected_count']
            or item['full_count'] != len(candidates)
            or np.any(np.diff(indices) <= 0)
            or not np.all(superclass[indices] == name)
            or not np.array_equal(indices, expected)
            or sha256_obj(obj=indices.tolist()) != item['indices_sha256']
            or body_ids != item['body_ids']
            or sha256_obj(obj=body_ids) != item['body_ids_sha256']
            or occupied.intersection(indices.tolist())
        ):
            raise ValueError(f'Frozen neuron mask differs from brain metadata: {name}.')
        occupied.update(indices.tolist())
        populations[name] = indices
    if not populations or len({len(indices) for indices in populations.values()}) != 1:
        raise ValueError('Observation masks must be nonempty and dimension matched.')
    return populations


def _checkpoints(case: TemporalCase) -> tuple[int, ...]:
    """Choose fixed stimulus and recovery sample steps for one case.

    :param case: Frozen condition timing.
    :type case: TemporalCase
    :returns: Strictly increasing global step indices including rest step zero.
    :rtype: tuple[int, ...]
    """
    stimulus = (0, 1, 2, 5, 10, 20) if case.stimulus_steps == 20 else (0, 1, 2, 5, 10)
    recovery = tuple(case.stimulus_steps + step for step in (1, 2, 5, 10))
    return (*stimulus, *recovery)


def _write_arrays(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    """Write deterministic compressed NPZ members without ever accepting pickle.

    :param path: New numeric evidence archive.
    :type path: Path
    :param arrays: Named numeric arrays, each with a stable shape and dtype.
    :type arrays: dict[str, NDArray[Any]]
    :raises ValueError: If any name or dtype is unsuitable for portable numeric evidence.
    """
    with ZipFile(file=path, mode='x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, values in sorted(arrays.items()):
            if not name.isidentifier() or values.dtype.kind not in 'biuf':
                raise ValueError('Response arrays require identifier names and numeric dtypes.')
            buffer = BytesIO()
            np.save(file=buffer, arr=values, allow_pickle=False)
            archive.writestr(zinfo_or_arcname=f'{name}.npy', data=buffer.getvalue())


def _check_parity(
    brain: EpisodeBrain,
    rest: RestState,
    sample_key: str,
    seed: int,
    input_idx: NDArray[np.int64],
    currents: NDArray[np.float32],
    response: TemporalResponse,
    stimulus_steps: int,
) -> None:
    """Replay the original runtime at the stimulus endpoint without changing evidence.

    :param brain: Recorder runtime with existing fly equations.
    :type brain: EpisodeBrain
    :param rest: Identical warmed initial state.
    :type rest: RestState
    :param sample_key: Noise-stream key for one isolated image window.
    :type sample_key: str
    :param seed: Parent experiment seed.
    :type seed: int
    :param input_idx: Encoder neuron indices (I,), int64.
    :type input_idx: NDArray[np.int64]
    :param currents: Original float32 kicks (I,1).
    :type currents: NDArray[np.float32]
    :param response: Instrumented checkpoint arrays.
    :type response: TemporalResponse
    :param stimulus_steps: Original stimulus duration in steps.
    :type stimulus_steps: int
    :raises ValueError: If independently replayed terminal features differ.
    """
    brain.begin(rest=rest, sample_ids=[sample_key], seed=seed)
    brain.run(input_idx=input_idx, currents=currents, n_steps=stimulus_steps)
    slot = response.checkpoints.index(stimulus_steps)
    expected = brain.features(kinds=['spike_trace', 'voltage'])
    trace = response.readout_traces[slot]
    voltage = response.voltages['descending_neuron'][slot]
    if not np.array_equal(expected[:, : trace.shape[1]], trace) or not np.array_equal(
        expected[:, trace.shape[1] :], voltage
    ):
        raise ValueError('Recorded terminal features differ from the existing runtime.')


def run_temporal_case(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    case_id: str,
    cohort_path: Path,
    masks_path: Path,
    membership_path: Path,
) -> dict[str, Any]:
    """Run one immutable training-only pilot condition and save numeric evidence.

    :param cfg: Original 20-identity effective experiment settings.
    :type cfg: ExperimentConfig
    :param paths: Working data, cache, and run directories.
    :type paths: Paths
    :param output: New attempt directory within the data home.
    :type output: Path
    :param case_id: One frozen C0-C6 condition identifier.
    :type case_id: str
    :param cohort_path: Committed exact training-member and image-hash document.
    :type cohort_path: Path
    :param masks_path: Committed graph source and population-mask audit.
    :type masks_path: Path
    :param membership_path: Archived parent development membership.
    :type membership_path: Path
    :returns: Completed attempt summary and response checksum.
    :rtype: dict[str, Any]
    :raises ValueError: If the protocol, source inputs, or reference parity fails.
    """
    if case_id not in CASES:
        raise ValueError('Unknown temporal-response condition.')
    case = CASES[case_id]
    if (
        cfg.encoder.amplitude != 0.05
        or cfg.brain.dt_s != 0.02
        or cfg.brain.steps_per_observation != 10
        or cfg.brain.warmup_steps != 25
        or cfg.brain.threads != 4
        or cfg.brain.batch_size != 1
        or cfg.readout.population != 'descending_neuron'
        or cfg.seed != 0
    ):
        raise ValueError('Source configuration differs from the frozen temporal protocol.')
    cohort = _read_json(path=cohort_path)
    masks = _read_json(path=masks_path)
    membership = _read_json(path=membership_path)
    parameters = {
        'kind': 'temporal_response',
        'case': case.name,
        'hypothesis': case.hypothesis,
        'stimulus_steps': case.stimulus_steps,
        'recovery_steps': case.recovery_steps,
        'current_scale': case.current_scale,
        'noise_enabled': case.noise_enabled,
        'cohort_sha256': sha256_file(path=cohort_path),
        'masks_sha256': sha256_file(path=masks_path),
        'membership_sha256': sha256_file(path=membership_path),
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        selected = _verify_cohort(
            cfg=cfg, paths=paths, prepared=prepared, cohort=cohort, membership=membership
        )
        populations = _verify_masks(paths=paths, document=masks)
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        original_threads = numba.get_num_threads()
        started = perf_counter()
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                warmup = EpisodeBrain(
                    brain_dir=paths.brain,
                    brain_cfg=cfg.brain,
                    readout_cfg=cfg.readout,
                    batch_size=1,
                    threads=cfg.brain.threads,
                )
                rest = warmup.compute_rest_state(seed=cfg.seed)
                if case.noise_enabled:
                    brain = warmup
                else:
                    del warmup
                    noise_off = cfg.brain.noise.model_copy(update={'enabled': False})
                    brain_cfg = cfg.brain.model_copy(update={'noise': noise_off})
                    brain = EpisodeBrain(
                        brain_dir=paths.brain,
                        brain_cfg=brain_cfg,
                        readout_cfg=cfg.readout,
                        batch_size=1,
                        threads=cfg.brain.threads,
                    )
                encoder = SparseProjectionEncoder(
                    cfg=cfg.encoder,
                    window=cfg.episodes.window,
                    candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
                )
                checkpoints = _checkpoints(case=case)
                rows: list[dict[str, Any]] = []
                recorded: dict[str, list[NDArray[Any]]] = {}
                for row_index, item in selected:
                    episode = builder.build(
                        sample=prepared.samples[row_index], image=prepared.images[row_index]
                    )
                    for window in cohort['windows']:
                        key = f'{episode.sample_id}|window={window}'
                        base = encoder.encode(
                            observations=episode.observations[window : window + 1],
                            positions=episode.positions[window : window + 1],
                        )
                        currents = (
                            None
                            if case.current_scale == 0
                            else (base * np.float32(case.current_scale)).astype(np.float32)
                        )
                        brain.begin(rest=rest, sample_ids=[key], seed=cfg.seed)
                        response = brain.run_recorded(
                            input_idx=encoder.input_idx,
                            currents=currents,
                            stimulus_steps=case.stimulus_steps,
                            recovery_steps=case.recovery_steps,
                            populations=populations,
                            checkpoints=checkpoints,
                        )
                        if case.name == 'C1':
                            assert currents is not None
                            _check_parity(
                                brain=brain,
                                rest=rest,
                                sample_key=key,
                                seed=cfg.seed,
                                input_idx=encoder.input_idx,
                                currents=currents,
                                response=response,
                                stimulus_steps=case.stimulus_steps,
                            )
                        for name in populations:
                            recorded.setdefault(f'voltage_{name}', []).append(
                                response.voltages[name][:, 0]
                            )
                            recorded.setdefault(f'spikes_{name}', []).append(
                                response.spike_counts[name][:, 0]
                            )
                        for name, values in (
                            ('readout_trace', response.readout_traces[:, 0]),
                            ('noise_kicks', response.noise_kicks[:, 0]),
                            ('total_spikes', response.total_spikes[:, 0]),
                            ('active_neurons', response.active_neurons[:, 0]),
                        ):
                            recorded.setdefault(name, []).append(values)
                        rows.append(
                            {
                                'sample_id': episode.sample_id,
                                'label': item['label'],
                                'window': window,
                            }
                        )
                        if (
                            psutil.Process().memory_info().rss > RESOURCE_LIMIT_BYTES
                            or perf_counter() - started > CASE_TIME_LIMIT_SECONDS
                        ):
                            raise RuntimeError('Temporal pilot resource limit exceeded.')
                arrays = {name: np.stack(values) for name, values in recorded.items()}
                _write_arrays(path=directory / 'responses.npz', arrays=arrays)
                write_json(path=directory / 'sample-windows.json', value=rows)
                if not case.noise_enabled and np.any(arrays['noise_kicks']):
                    raise ValueError('Episode noise was present in a noise-off condition.')
                report = {
                    'schema_version': 1,
                    'case': case.name,
                    'checkpoints': checkpoints,
                    'sample_windows': len(rows),
                    'populations': {name: len(indices) for name, indices in populations.items()},
                    'dt_s': cfg.brain.dt_s,
                    'stimulus_steps': case.stimulus_steps,
                    'current_scale': case.current_scale,
                    'noise_enabled': case.noise_enabled,
                    'encoder': encoder.spec(),
                    'response_sha256': sha256_file(path=directory / 'responses.npz'),
                    'total_spikes': int(arrays['total_spikes'].sum()),
                    'noise_kicks': int(arrays['noise_kicks'].sum()),
                    'conclusion': (
                        f'Recorded {len(rows)} isolated training-window responses for {case.name}; '
                        'paired scientific analysis is pending.'
                    ),
                }
                write_json(path=directory / 'report.json', value=report)
        finally:
            numba.set_num_threads(n=original_threads)
    return {'case': case.name, 'status': 'completed', 'output': str(directory), **report}

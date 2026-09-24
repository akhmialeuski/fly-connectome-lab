"""Source-verified, training-only identity geometry in frozen fly responses."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

from flystate.brain.runtime import EpisodeBrain
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.identity import CASES, _check_schedule
from flystate.diagnostics.temporal import _read_json, _verify_cohort, _verify_masks
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, stable_int
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table


@dataclass(frozen=True)
class IdentityEvidence:
    """Verified one-case numeric responses and paired blank arrays."""

    case_id: str
    rows: list[dict[str, Any]]
    driven: dict[str, NDArray[Any]]
    blank: dict[str, NDArray[Any]]
    inventory_sha256: str


def _load_case(
    paths: Paths,
    source: Path,
    case_id: str,
    cfg: ExperimentConfig,
    schedule: dict[str, Any],
    cohort_sha256: str,
    masks_sha256: str,
    membership_sha256: str,
    schedule_sha256: str,
    expected_populations: dict[str, int],
    readout_width: int,
) -> IdentityEvidence:
    """Verify an immutable case inventory, schema, numeric arrays, and source identity.

    :param paths: Working data home.
    :type paths: Paths
    :param source: Parent of the three completed case attempts.
    :type source: Path
    :param case_id: Frozen N0, N1, or OFF identifier.
    :type case_id: str
    :param cfg: Effective original experiment configuration.
    :type cfg: ExperimentConfig
    :param schedule: Frozen timing, case, and mask order.
    :type schedule: dict[str, Any]
    :param cohort_sha256: Exact committed cohort file digest.
    :type cohort_sha256: str
    :param masks_sha256: Exact committed mask file digest.
    :type masks_sha256: str
    :param membership_sha256: Exact committed membership file digest.
    :type membership_sha256: str
    :param schedule_sha256: Exact committed schedule file digest.
    :type schedule_sha256: str
    :param expected_populations: Frozen selected-neuron counts by mask.
    :type expected_populations: dict[str, int]
    :param readout_width: Full descending readout population size.
    :type readout_width: int
    :returns: Verified rows and numeric arrays for this case.
    :rtype: IdentityEvidence
    :raises ValueError: If provenance, array schema, or numerical evidence differs.
    """
    directory = source / case_id
    manifest = verify_attempt_inventory(directory=directory, paths=paths)
    parameters = manifest['parameters']
    if (
        manifest['status'] != 'completed'
        or parameters['kind'] != 'identity_access'
        or parameters['case'] != case_id
        or parameters['noise_enabled'] != CASES[case_id]['noise_enabled']
        or parameters['episode_seed'] != CASES[case_id]['episode_seed']
        or parameters['cohort_sha256'] != cohort_sha256
        or parameters['masks_sha256'] != masks_sha256
        or parameters['membership_sha256'] != membership_sha256
        or parameters['schedule_sha256'] != schedule_sha256
        or json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        != cfg.model_dump(mode='json')
    ):
        raise ValueError(f'Case {case_id} has incompatible frozen source provenance.')
    report = _read_json(path=directory / 'report.json')
    rows = json.loads((directory / 'sample-windows.json').read_text(encoding='utf-8'))
    if (
        report['case'] != case_id
        or report['stimulus_steps'] != schedule['stimulus_steps']
        or report['recovery_steps'] != schedule['recovery_steps']
        or report['checkpoints'] != schedule['checkpoints']
        or report['windows'] != schedule['windows']
        or report['sample_windows'] != len(rows)
        or report['blank_windows'] != len(schedule['windows'])
        or report['response_sha256'] != sha256_file(path=directory / 'responses.npz')
        or report['blank_sha256'] != sha256_file(path=directory / 'blanks.npz')
        or report['populations'] != expected_populations
    ):
        raise ValueError(f'Case {case_id} has incompatible response metadata.')
    with (
        np.load(file=directory / 'responses.npz', allow_pickle=False) as response_source,
        np.load(file=directory / 'blanks.npz', allow_pickle=False) as blank_source,
    ):
        driven = {name: response_source[name].copy() for name in response_source.files}
        blank = {name: blank_source[name].copy() for name in blank_source.files}
    names = {
        *(f'voltage_{name}' for name in schedule['masks_order']),
        *(f'spikes_{name}' for name in schedule['masks_order']),
        'readout_trace',
        'noise_kicks',
        'noise_digests',
        'total_spikes',
        'active_neurons',
    }
    if set(driven) != names or set(blank) != names:
        raise ValueError(f'Case {case_id} has missing or unexpected response arrays.')
    checkpoints = len(schedule['checkpoints'])
    steps = schedule['stimulus_steps'] + schedule['recovery_steps']
    for arrays, count in ((driven, len(rows)), (blank, len(schedule['windows']))):
        for name in schedule['masks_order']:
            width = report['populations'][name]
            voltage = arrays[f'voltage_{name}']
            spikes = arrays[f'spikes_{name}']
            if (
                voltage.shape != (count, checkpoints, width)
                or voltage.dtype != np.float32
                or not np.isfinite(voltage).all()
                or spikes.shape != voltage.shape
                or spikes.dtype != np.int32
                or np.any(spikes < 0)
                or np.any(np.diff(spikes, axis=1) < 0)
                or np.any(spikes[:, 0] != 0)
            ):
                raise ValueError(f'Case {case_id} has invalid {name} trajectories.')
        traces = arrays['readout_trace']
        if (
            traces.shape != (count, checkpoints, readout_width)
            or traces.dtype != np.float32
            or not np.isfinite(traces).all()
        ):
            raise ValueError(f'Case {case_id} has invalid readout traces.')
        for name in ('noise_kicks', 'total_spikes', 'active_neurons'):
            values = arrays[name]
            if values.shape != (count, steps) or values.dtype != np.int64 or np.any(values < 0):
                raise ValueError(f'Case {case_id} has invalid per-step {name}.')
        digests = arrays['noise_digests']
        if digests.shape != (count, steps, 32) or digests.dtype != np.uint8:
            raise ValueError(f'Case {case_id} has invalid noise-index digests.')
        if np.any(np.diff(arrays['active_neurons'], axis=1) < 0):
            raise ValueError(f'Case {case_id} has decreasing active-neuron counts.')
    return IdentityEvidence(
        case_id=case_id,
        rows=rows,
        driven=driven,
        blank=blank,
        inventory_sha256=sha256_file(path=directory / 'checksums.sha256'),
    )


def _verify_controls(
    records: dict[str, IdentityEvidence],
    cohort: dict[str, Any],
    schedule: dict[str, Any],
) -> dict[str, Any]:
    """Require ordered membership and exact image-independent noise index controls.

    :param records: Three verified condition attempts.
    :type records: dict[str, IdentityEvidence]
    :param cohort: Frozen two-photo-per-identity selection.
    :type cohort: dict[str, Any]
    :param schedule: Frozen windows and population order.
    :type schedule: dict[str, Any]
    :returns: Source and blank-control verification summary.
    :rtype: dict[str, Any]
    :raises ValueError: If order, noise indices, rest states, or OFF control fail.
    """
    expected = [
        (item['sample_id'], item['label'], window)
        for item in cohort['samples']
        for window in schedule['windows']
    ]
    for case_id, record in records.items():
        observed = [(row['sample_id'], row['label'], row['window']) for row in record.rows]
        if observed != expected or record.rows != records['N0'].rows:
            raise ValueError(f'Case {case_id} changed frozen sample/window membership.')
        for row_index, (_, _, window) in enumerate(expected):
            blank_slot = schedule['windows'].index(window)
            for name in schedule['masks_order']:
                if not np.array_equal(
                    record.driven[f'voltage_{name}'][row_index, 0],
                    record.blank[f'voltage_{name}'][blank_slot, 0],
                ):
                    raise ValueError(f'Case {case_id} changed the paired rest voltage.')
            if not np.array_equal(
                record.driven['readout_trace'][row_index, 0],
                record.blank['readout_trace'][blank_slot, 0],
            ):
                raise ValueError(f'Case {case_id} changed the paired rest trace.')
            for name in ('noise_kicks', 'noise_digests'):
                if not np.array_equal(
                    record.driven[name][row_index], record.blank[name][blank_slot]
                ):
                    raise ValueError(f'Case {case_id} changed the paired noise indices.')
    empty_digest = np.frombuffer(buffer=hashlib.sha256(b'').digest(), dtype=np.uint8)
    if (
        not np.any(records['N0'].driven['noise_kicks'])
        or not np.any(records['N1'].driven['noise_kicks'])
        or np.array_equal(
            records['N0'].driven['noise_digests'], records['N1'].driven['noise_digests']
        )
        or np.any(records['OFF'].driven['noise_kicks'])
        or np.any(records['OFF'].blank['noise_kicks'])
        or not np.all(records['OFF'].driven['noise_digests'] == empty_digest)
        or not np.all(records['OFF'].blank['noise_digests'] == empty_digest)
    ):
        raise ValueError('Identity-access noise-on/off controls did not replay.')
    return {
        'cases_verified': len(records),
        'images_verified': len(cohort['samples']),
        'driven_windows_per_case': len(expected),
        'blank_windows_per_case': len(schedule['windows']),
        'ordered_membership_exact': True,
        'paired_rest_exact': True,
        'paired_noise_indices_exact': True,
        'noise_streams_distinct': True,
        'noise_off_zero': True,
    }


def _input_features(
    cfg: ExperimentConfig,
    paths: Paths,
    cohort: dict[str, Any],
    masks: dict[str, Any],
    schedule: dict[str, Any],
    rows: list[dict[str, Any]],
    membership: dict[str, Any],
) -> dict[str, NDArray[np.float64]]:
    """Recompute pixel and encoded-current controls without archiving images.

    :param cfg: Source experiment configuration and encoder seed.
    :type cfg: ExperimentConfig
    :param paths: Registered external image and graph sources.
    :type paths: Paths
    :param cohort: Frozen exact image selection and SHA-256 values.
    :type cohort: dict[str, Any]
    :param masks: Frozen graph and observation-mask source document.
    :type masks: dict[str, Any]
    :param schedule: Fixed three raster windows.
    :type schedule: dict[str, Any]
    :param rows: Verified per-image/window input hashes and current norms.
    :type rows: list[dict[str, Any]]
    :param membership: Original train/validation membership document.
    :type membership: dict[str, Any]
    :returns: Pixel and encoded-current matrices of shape (40,F), float64.
    :rtype: dict[str, NDArray[np.float64]]
    :raises ValueError: If a source image, pixel window, or encoded current differs.
    """
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    selected = _verify_cohort(
        cfg=cfg, paths=paths, prepared=prepared, cohort=cohort, membership=membership
    )
    _verify_masks(paths=paths, document=masks)
    threads = cfg.brain.threads
    if threads is None:
        raise ValueError('Identity analysis requires an explicit Numba thread count.')
    original_threads = numba.get_num_threads()
    try:
        brain = EpisodeBrain(
            brain_dir=paths.brain,
            brain_cfg=cfg.brain,
            readout_cfg=cfg.readout,
            batch_size=1,
            threads=threads,
        )
        encoder = SparseProjectionEncoder(
            cfg=cfg.encoder,
            window=cfg.episodes.window,
            candidate_neurons=brain.cells(superclasses=[cfg.encoder.target_population]),
        )
    finally:
        numba.set_num_threads(n=original_threads)
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    pixel_vectors: list[NDArray[np.float64]] = []
    current_vectors: list[NDArray[np.float64]] = []
    position = 0
    for row_index, _ in selected:
        episode = builder.build(
            sample=prepared.samples[row_index], image=prepared.images[row_index]
        )
        pixel_blocks: list[NDArray[np.float64]] = []
        current_blocks: list[NDArray[np.float64]] = []
        for window in schedule['windows']:
            observation = episode.observations[window : window + 1]
            currents = encoder.encode(
                observations=observation,
                positions=episode.positions[window : window + 1],
            )
            expected = rows[position]
            if (
                expected['pixel_sha256'] != hashlib.sha256(observation.tobytes()).hexdigest()
                or expected['current_sha256'] != hashlib.sha256(currents.tobytes()).hexdigest()
                or expected['current_nonzero'] != int(np.count_nonzero(a=currents))
                or not np.isclose(
                    a=expected['current_l2'],
                    b=np.sqrt(np.sum(a=currents.astype(np.float64) ** 2, dtype=np.float64)),
                    atol=0,
                    rtol=1e-14,
                )
            ):
                raise ValueError('Recorded input digest or current magnitude changed.')
            pixel_blocks.append(observation.reshape(-1).astype(np.float64))
            current_blocks.append(currents.reshape(-1).astype(np.float64))
            position += 1
        pixel_vectors.append(np.concatenate(pixel_blocks))
        current_vectors.append(np.concatenate(current_blocks))
    return {
        'pixels': np.stack(pixel_vectors),
        'encoded_current': np.stack(current_vectors),
    }


def _evoked_matrix(
    record: IdentityEvidence,
    schedule: dict[str, Any],
    population: str,
    window: int,
    step: int,
) -> NDArray[np.float64]:
    """Return paired voltage vectors for one fixed population/window/checkpoint.

    :param record: Verified case evidence with (image,window) row order.
    :type record: IdentityEvidence
    :param schedule: Frozen windows and checkpoint positions.
    :type schedule: dict[str, Any]
    :param population: One frozen anatomical mask name.
    :type population: str
    :param window: One frozen raster window index.
    :type window: int
    :param step: Global physical checkpoint in 20 ms units.
    :type step: int
    :returns: Image-by-neuron float64 evoked voltage matrix, shape (40,1314) in real data.
    :rtype: NDArray[np.float64]
    """
    window_slot = schedule['windows'].index(window)
    checkpoint_slot = schedule['checkpoints'].index(step)
    stride = len(schedule['windows'])
    name = f'voltage_{population}'
    driven = record.driven[name][window_slot::stride, checkpoint_slot].astype(np.float64)
    blank = record.blank[name][window_slot, checkpoint_slot].astype(np.float64)
    return driven - blank


def _primary_matrix(record: IdentityEvidence, schedule: dict[str, Any]) -> NDArray[np.float64]:
    """Concatenate all fixed masks, then windows, at the original stimulus endpoint.

    :param record: Verified case evidence.
    :type record: IdentityEvidence
    :param schedule: Frozen mask/window ordering and primary checkpoint.
    :type schedule: dict[str, Any]
    :returns: Image-by-feature float64 matrix, (40,15768) for real data.
    :rtype: NDArray[np.float64]
    """
    return np.concatenate(
        [
            _evoked_matrix(
                record=record,
                schedule=schedule,
                population=population,
                window=window,
                step=schedule['primary_endpoint_step'],
            )
            for population in schedule['masks_order']
            for window in schedule['windows']
        ],
        axis=1,
    )


def _view_metrics(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    sample_ids: list[str],
    case_id: str,
    view: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], NDArray[np.int64]]:
    """Calculate every fixed pair distance and two-direction 1-NN prediction.

    :param features: Finite image-by-feature float64 matrix, shape (2C,F).
    :type features: NDArray[np.float64]
    :param labels: Two consecutive rows per identity, shape (2C,), int64.
    :type labels: NDArray[np.int64]
    :param sample_ids: Ordered stable image IDs corresponding to rows.
    :type sample_ids: list[str]
    :param case_id: N0, N1, OFF, or INPUT.
    :type case_id: str
    :param view: Predeclared primary, input, or diagnostic feature name.
    :type view: str
    :returns: Summary, full pair rows, 1-NN query rows, and predictions (2,C).
    :rtype: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], NDArray[np.int64]]
    :raises ValueError: If rows, labels, or numeric vectors are invalid.
    """
    if (
        features.ndim != 2
        or features.shape[0] != len(labels)
        or features.shape[0] != len(sample_ids)
        or features.shape[0] % 2
        or not np.isfinite(features).all()
        or labels.tolist() != [label for label in range(len(labels) // 2) for _ in range(2)]
    ):
        raise ValueError('Identity view requires ordered, finite two-photo-per-label rows.')
    pair_rows: list[dict[str, Any]] = []
    within: list[float] = []
    between: list[float] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            difference = features[first] - features[second]
            distance = float(np.sum(a=difference * difference, dtype=np.float64))
            same = bool(labels[first] == labels[second])
            (within if same else between).append(distance)
            pair_rows.append(
                {
                    'case': case_id,
                    'view': view,
                    'first_sample_id': sample_ids[first],
                    'second_sample_id': sample_ids[second],
                    'same_identity': same,
                    'squared_distance': distance,
                }
            )
    predictions: list[NDArray[np.int64]] = []
    query_rows: list[dict[str, Any]] = []
    for direction, (references, queries) in enumerate(
        ((features[::2], features[1::2]), (features[1::2], features[::2]))
    ):
        difference = queries[:, None, :] - references[None, :, :]
        distances = np.sum(a=difference * difference, axis=2, dtype=np.float64)
        predicted = np.argmin(a=distances, axis=1).astype(np.int64)
        predictions.append(predicted)
        for label, guess in enumerate(predicted):
            query_rows.append(
                {
                    'case': case_id,
                    'view': view,
                    'direction': direction,
                    'sample_id': sample_ids[2 * label + (1 - direction)],
                    'true_label': label,
                    'predicted_label': int(guess),
                    'correct': bool(guess == label),
                    'squared_distance': float(distances[label, guess]),
                }
            )
    mean_between = float(np.mean(a=between))
    summary = {
        'features': features.shape[1],
        'within_pairs': len(within),
        'between_pairs': len(between),
        'within_mean_squared_distance': float(np.mean(a=within)),
        'within_median_squared_distance': float(np.median(a=within)),
        'between_mean_squared_distance': mean_between,
        'between_median_squared_distance': float(np.median(a=between)),
        'within_between_mean_ratio': float(np.mean(a=within) / mean_between)
        if mean_between > 0
        else None,
        'correct': int(sum(row['correct'] for row in query_rows)),
        'queries': len(query_rows),
        'accuracy': float(np.mean(a=[row['correct'] for row in query_rows])),
    }
    return summary, pair_rows, query_rows, np.stack(predictions)


def _permutation_test(predictions: NDArray[np.int64], draws: int, key: str) -> dict[str, Any]:
    """Test the fixed 40-query score against seeded identity-label permutations.

    :param predictions: Two directions by C predicted identity labels, int64.
    :type predictions: NDArray[np.int64]
    :param draws: Frozen number of identity-level random permutations.
    :type draws: int
    :param key: Frozen stable seed-derivation key.
    :type key: str
    :returns: Observed score, complete null histogram, and plus-one p-value.
    :rtype: dict[str, Any]
    """
    labels = np.arange(predictions.shape[1], dtype=np.int64)
    observed = int(np.sum(a=predictions == labels[None, :]))
    generator = np.random.default_rng(seed=np.random.SeedSequence(entropy=[0, stable_int(key=key)]))
    histogram: dict[int, int] = {}
    extreme = 0
    for _ in range(draws):
        permutation = generator.permutation(labels)
        score = int(np.sum(a=predictions == permutation[None, :]))
        histogram[score] = histogram.get(score, 0) + 1
        extreme += score >= observed
    return {
        'observed_correct': observed,
        'draws': draws,
        'seed_key': key,
        'null_score_histogram': {str(score): count for score, count in sorted(histogram.items())},
        'upper_tail_p': (1 + extreme) / (draws + 1),
    }


def analyze_identity_access(
    cfg: ExperimentConfig,
    paths: Paths,
    source: Path,
    output: Path,
    cohort_path: Path,
    masks_path: Path,
    membership_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    """Verify all T28 sources and compute the frozen identity-access analysis.

    :param cfg: Original effective experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home with registered external sources.
    :type paths: Paths
    :param source: Parent directory of N0, N1, and OFF attempts.
    :type source: Path
    :param output: Fresh immutable analysis attempt under FLYSTATE_HOME.
    :type output: Path
    :param cohort_path: Frozen selected-image protocol file.
    :type cohort_path: Path
    :param masks_path: Frozen anatomical observation masks.
    :type masks_path: Path
    :param membership_path: Original training membership file.
    :type membership_path: Path
    :param schedule_path: Frozen response and analysis schedule.
    :type schedule_path: Path
    :returns: Control gate, primary/secondary measurements, and attempt path.
    :rtype: dict[str, Any]
    :raises ValueError: If any source, attempt, or matched control is incompatible.
    """
    source_dir = output_path(path=source, paths=paths)
    cohort = _read_json(path=cohort_path)
    masks = _read_json(path=masks_path)
    membership = _read_json(path=membership_path)
    schedule = _read_json(path=schedule_path)
    _check_schedule(
        cfg=cfg,
        schedule=schedule,
        cohort_path=cohort_path,
        masks_path=masks_path,
        membership_path=membership_path,
    )
    if (
        config_hash(cfg=cfg) != cohort['source_config_sha256']
        or cohort['source_pilot_masks_sha256'] != schedule['masks_sha256']
        or cohort['windows'] != schedule['windows']
        or len(cohort['samples']) != 2 * cfg.dataset.subset.n_identities
    ):
        raise ValueError('Analysis inputs differ from the frozen identity-access cohort.')
    parameters = {
        'kind': 'identity_access_analysis',
        'hypothesis': 'Measure same-identity access in original-current fly responses.',
        'source': str(source_dir.relative_to(paths.home)),
        'cohort_sha256': sha256_file(path=cohort_path),
        'masks_sha256': sha256_file(path=masks_path),
        'membership_sha256': sha256_file(path=membership_path),
        'schedule_sha256': sha256_file(path=schedule_path),
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        records = {
            case_id: _load_case(
                paths=paths,
                source=source_dir,
                case_id=case_id,
                cfg=cfg,
                schedule=schedule,
                cohort_sha256=parameters['cohort_sha256'],
                masks_sha256=parameters['masks_sha256'],
                membership_sha256=parameters['membership_sha256'],
                schedule_sha256=parameters['schedule_sha256'],
                expected_populations={
                    name: masks['masks'][name]['selected_count'] for name in schedule['masks_order']
                },
                readout_width=masks['masks']['descending_neuron']['full_count'],
            )
            for case_id in CASES
        }
        controls = _verify_controls(records=records, cohort=cohort, schedule=schedule)
        input_matrices = _input_features(
            cfg=cfg,
            paths=paths,
            cohort=cohort,
            masks=masks,
            schedule=schedule,
            rows=records['N0'].rows,
            membership=membership,
        )
        controls['input_digests_exact'] = True
        labels = np.asarray(a=[item['label'] for item in cohort['samples']], dtype=np.int64)
        sample_ids = [item['sample_id'] for item in cohort['samples']]
        pair_rows: list[dict[str, Any]] = []
        query_rows: list[dict[str, Any]] = []
        primary: dict[str, Any] = {}
        secondary: list[dict[str, Any]] = []
        primary_matrices: dict[str, NDArray[np.float64]] = {}
        for case_id, record in records.items():
            primary_matrix = _primary_matrix(record=record, schedule=schedule)
            primary_matrices[case_id] = primary_matrix
            summary, pairs, queries, predictions = _view_metrics(
                features=primary_matrix,
                labels=labels,
                sample_ids=sample_ids,
                case_id=case_id,
                view='all_masks_all_windows_step10',
            )
            summary['permutation'] = _permutation_test(
                predictions=predictions,
                draws=schedule['null_permutations'],
                key=schedule['null_seed_key'],
            )
            primary[case_id] = summary
            pair_rows.extend(pairs)
            query_rows.extend(queries)
            for population in schedule['masks_order']:
                for window in schedule['windows']:
                    for step in (10, 20):
                        view = f'{population}|window={window}|step={step}'
                        feature = _evoked_matrix(
                            record=record,
                            schedule=schedule,
                            population=population,
                            window=window,
                            step=step,
                        )
                        details, pairs, queries, _ = _view_metrics(
                            features=feature,
                            labels=labels,
                            sample_ids=sample_ids,
                            case_id=case_id,
                            view=view,
                        )
                        secondary.append({'case': case_id, 'view': view, **details})
                        pair_rows.extend(pairs)
                        query_rows.extend(queries)
        input_controls: dict[str, Any] = {}
        for name, values in input_matrices.items():
            summary, pairs, queries, _ = _view_metrics(
                features=values,
                labels=labels,
                sample_ids=sample_ids,
                case_id='INPUT',
                view=name,
            )
            input_controls[name] = summary
            pair_rows.extend(pairs)
            query_rows.extend(queries)
        delta = primary_matrices['N0'] - primary_matrices['N1']
        same_image_cross_seed = np.sum(a=delta * delta, axis=1, dtype=np.float64)
        cross_seed_rows = [
            {
                'sample_id': sample_id,
                'label': int(label),
                'squared_distance': float(distance),
            }
            for sample_id, label, distance in zip(
                sample_ids, labels, same_image_cross_seed, strict=True
            )
        ]
        gate = all(
            primary[case_id]['permutation']['upper_tail_p'] < 0.05
            and primary[case_id]['within_between_mean_ratio'] is not None
            and primary[case_id]['within_between_mean_ratio'] < 1
            for case_id in ('N0', 'N1')
        )
        write_table(path=directory / 'pairs.parquet', rows=pair_rows)
        write_table(path=directory / 'queries.parquet', rows=query_rows)
        write_table(path=directory / 'cross-seed.parquet', rows=cross_seed_rows)
        report = {
            'schema_version': 1,
            'controls': controls,
            'primary': primary,
            'input_controls': input_controls,
            'secondary': secondary,
            'same_image_cross_seed_mean_squared_distance': float(np.mean(a=same_image_cross_seed)),
            'same_image_cross_seed_median_squared_distance': float(
                np.median(a=same_image_cross_seed)
            ),
            'pair_rows': len(pair_rows),
            'query_rows': len(query_rows),
            'cross_seed_rows': len(cross_seed_rows),
            'chance_accuracy': 1 / cfg.dataset.subset.n_identities,
            'gate_passed': gate,
            'decision': 'freeze_fixed_connectome_recognition_test'
            if gate
            else 'investigate_signal_geometry_before_dynamics',
            'source_inventories_sha256': {
                case_id: record.inventory_sha256 for case_id, record in records.items()
            },
            'conclusion': (
                'The fixed primary training-only gate passed in both original-noise streams.'
                if gate
                else 'The fixed primary training-only gate failed in at least one noise stream.'
            )
            + ' This is not validation accuracy, untouched confirmation, or sequential memory.',
        }
        write_json(path=directory / 'report.json', value=report)
    return {'status': 'completed', 'output': str(directory), **report}

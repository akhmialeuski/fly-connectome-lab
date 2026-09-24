"""Frozen training-only pixel and encoded-current identity access study."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.preprocess import PreparedDataset, prepare_dataset
from flystate.datasets.registry import require_validated
from flystate.diagnostics.artifacts import attempt, export_classifier
from flystate.diagnostics.probes import score_split
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, sha256_obj, stable_int
from flystate.readouts.fitting import fit_classifier
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

REPRESENTATIONS: tuple[str, ...] = ('pixels', 'encoded_current')


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON protocol object.

    :param path: Existing UTF-8 protocol file.
    :type path: Path
    :returns: Parsed mapping.
    :rtype: dict[str, Any]
    :raises ValueError: If the JSON root is not an object.
    """
    value = json.loads(s=path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Input-access protocol must be a JSON object.')
    return value


def _verify_protocol(
    cfg: ExperimentConfig,
    paths: Paths,
    prepared: PreparedDataset,
    cohort: dict[str, Any],
    schedule: dict[str, Any],
    membership: dict[str, Any],
    membership_path: Path,
    cohort_path: Path,
) -> list[int]:
    """Verify all 280 source images and the frozen fit/query and CV assignments.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: External dataset and brain paths.
    :type paths: Paths
    :param prepared: Content-verified aligned uint8 RGB cache, shape (400,128,128,3).
    :type prepared: PreparedDataset
    :param cohort: Frozen source hashes, labels, roles, and ranks.
    :type cohort: dict[str, Any]
    :param schedule: Frozen two-representation CV and gate schedule.
    :type schedule: dict[str, Any]
    :param membership: Archived original split membership.
    :type membership: dict[str, Any]
    :param membership_path: Exact archived membership file.
    :type membership_path: Path
    :param cohort_path: Exact committed cohort file.
    :type cohort_path: Path
    :returns: Prepared-image row indices in frozen cohort order.
    :rtype: list[int]
    :raises ValueError: If a source byte, assignment, or fitting setting differs.
    """
    members = membership['sample_ids_by_split']
    if (
        sha256_obj(obj=members) != membership['membership_sha256']
        or cohort['source_membership_sha256'] != membership['membership_sha256']
        or cohort['source_membership_file_sha256'] != sha256_file(path=membership_path)
        or cohort['source_config_sha256'] != config_hash(cfg=cfg)
        or cohort['preprocess_cache_key'] != prepared.key
        or cohort['dataset_fingerprint'] != prepared.fingerprint
        or schedule['cohort_sha256'] != sha256_file(path=cohort_path)
        or schedule['representations'] != list(REPRESENTATIONS)
        or schedule['windows'] != list(range(cfg.episodes.steps))
        or schedule['pixel_scale_divisor'] != 255
    ):
        raise ValueError('Input-access source or protocol identity differs.')
    readout = schedule['readout']
    if (
        readout['preprocessing'] != ['StandardScaler', 'PCA']
        or readout['pca_components'] != cfg.readout.pca_components
        or readout['c_grid'] != cfg.readout.c_grid
        or readout['cv_folds'] != cfg.readout.cv_folds
        or readout['seed'] != cfg.seed
        or readout['logistic_tolerance'] != 1e-6
        or readout['max_iterations'] != 50000
        or readout['blas_threads'] != 1
        or cohort['fit_per_identity'] != 10
        or cohort['query_per_identity'] != 4
        or schedule['gate']
        != {
            'min_correct_of_80_each': 20,
            'max_log_loss_exclusive': 'ln(20)',
            'chance_accuracy': 0.05,
        }
    ):
        raise ValueError('Input-access readout or gate changed from the frozen protocol.')
    rows_by_id = {
        sample.sample_id: (index, sample) for index, sample in enumerate(prepared.samples)
    }
    if len(rows_by_id) != len(prepared.samples):
        raise ValueError('Selected source samples contain duplicate identifiers.')
    frozen = cohort['samples']
    if len(frozen) != 280 or len({item['sample_id'] for item in frozen}) != 280:
        raise ValueError('Input-access cohort must contain 280 unique training images.')
    image_root = (
        CelebAAdapter(root=require_validated(paths=paths, name=cfg.dataset.name), expected=None)
        .locate()
        .images_dir
    )
    ordered_rows: list[int] = []
    expected_items: list[tuple[str, int, int, str]] = []
    for label in range(20):
        ranked = sorted(
            (
                sample
                for sample in prepared.samples
                if sample.split == 'train' and sample.label == label
            ),
            key=lambda sample: (stable_int(key=sample.sample_id), sample.sample_id),
        )
        if len(ranked) != 14:
            raise ValueError(f'Identity {label} must have exactly 14 training photographs.')
        expected_items.extend(
            (sample.sample_id, label, rank, 'fit' if rank < 10 else 'query')
            for rank, sample in enumerate(ranked)
        )
    if [
        (item['sample_id'], item['label'], item['rank'], item['role']) for item in frozen
    ] != expected_items or set(item['sample_id'] for item in frozen) != set(members['train']):
        raise ValueError('Frozen fit/query assignment differs from the rank rule.')
    for item in frozen:
        row, sample = rows_by_id[item['sample_id']]
        if (
            sample.split != 'train'
            or sample.label != item['label']
            or sha256_file(path=image_root / sample.filename) != item['source_sha256']
            or hashlib.sha256(prepared.images[row].tobytes()).hexdigest() != item['aligned_sha256']
        ):
            raise ValueError(f'Input-access source image differs: {item["sample_id"]}.')
        ordered_rows.append(row)
    fit = [item for item in frozen if item['role'] == 'fit']
    fit_ids = [item['sample_id'] for item in fit]
    labels = np.asarray(a=[item['label'] for item in fit], dtype=np.int64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    expected_cv = [
        {
            'index': index,
            'fit_ids': [fit_ids[i] for i in training],
            'held_out_ids': [fit_ids[i] for i in held_out],
            'pca_components': cfg.readout.pca_components,
        }
        for index, (training, held_out) in enumerate(folds.split(X=np.arange(len(fit)), y=labels))
    ]
    if schedule['cv'] != expected_cv:
        raise ValueError('Frozen CV fold membership differs from the fitting algorithm.')
    return ordered_rows


def _features(
    cfg: ExperimentConfig, paths: Paths, prepared: PreparedDataset, rows: list[int]
) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
    """Recompute all-window pixel/current features without saving image-derived arrays.

    :param cfg: Original deterministic raster and sparse-encoder settings.
    :type cfg: ExperimentConfig
    :param paths: External brain metadata path.
    :type paths: Paths
    :param prepared: Read-only aligned uint8 RGB images, shape (400,128,128,3).
    :type prepared: PreparedDataset
    :param rows: The 280 verified original training image indices.
    :type rows: list[int]
    :returns: Two float32 matrices (280,F) and source/encoder hashes.
    :rtype: tuple[dict[str, NDArray[np.float32]], dict[str, Any]]
    :raises ValueError: If a representation is nonfinite or has an unexpected shape.
    """
    with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as brain:
        candidates = np.flatnonzero(
            a=(brain['superclass'] == cfg.encoder.target_population)
            | (brain['cell_type'] == cfg.encoder.target_population)
        )
    encoder = SparseProjectionEncoder(
        cfg=cfg.encoder, window=cfg.episodes.window, candidate_neurons=candidates
    )
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    pixel_width = cfg.episodes.steps * cfg.episodes.window**2 * 3
    current_width = cfg.episodes.steps * len(encoder.input_idx)
    pixels = np.empty(shape=(len(rows), pixel_width), dtype=np.float32)
    currents = np.empty(shape=(len(rows), current_width), dtype=np.float32)
    for offset, row in enumerate(rows):
        episode = builder.build(sample=prepared.samples[row], image=prepared.images[row])
        pixels[offset] = episode.observations.reshape(-1).astype(np.float32) / 255
        currents[offset] = encoder.encode(
            observations=episode.observations, positions=episode.positions
        ).T.reshape(-1)
    if not np.isfinite(pixels).all() or not np.isfinite(currents).all():
        raise ValueError('Input-access features contain nonfinite values.')
    features = {'pixels': pixels, 'encoded_current': currents}
    provenance = {
        'encoder': encoder.spec(),
        'brain_files_sha256': {
            name: sha256_file(path=paths.brain / name) for name in ('brain.npz', 'weights.npz')
        },
        'feature_shapes': {name: list(value.shape) for name, value in features.items()},
        'feature_sha256': {
            name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in features.items()
        },
        'source_images_or_feature_arrays_saved': False,
    }
    return features, provenance


def _summarize_queries(
    model: Any, features: NDArray[np.float32], labels: NDArray[np.int64]
) -> dict[str, Any]:
    """Describe fixed held-out training-query predictions and 10-bin reliability.

    :param model: Fitted sklearn pipeline with 20 sorted identity classes.
    :type model: Any
    :param features: Float32 query matrix (80,F).
    :type features: NDArray[np.float32]
    :param labels: Integer identity labels (80,).
    :type labels: NDArray[np.int64]
    :returns: Per-identity counts and fixed-bin confidence/accuracy records.
    :rtype: dict[str, Any]
    """
    probabilities = model.predict_proba(X=features.astype(np.float64))
    predicted = model.classes_[np.argmax(a=probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    bins: list[dict[str, Any]] = []
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = (confidence >= low) & (
            (confidence <= high) if index == 9 else (confidence < high)
        )
        bins.append(
            {
                'lower_inclusive': low,
                'upper_exclusive_except_last': high,
                'count': int(np.count_nonzero(a=selected)),
                'mean_confidence': float(np.mean(a=confidence[selected]))
                if selected.any()
                else None,
                'accuracy': float(np.mean(a=predicted[selected] == labels[selected]))
                if selected.any()
                else None,
            }
        )
    return {
        'per_identity_correct': {
            str(label): int(np.count_nonzero(a=(predicted == labels) & (labels == label)))
            for label in range(20)
        },
        'per_identity_total': dict(Counter(int(label) for label in labels)),
        'reliability_bins': bins,
    }


def run_input_access(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    cohort_path: Path,
    schedule_path: Path,
    membership_path: Path,
) -> dict[str, Any]:
    """Fit two frozen input probes and score only held-out training photographs.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home; generated files stay beneath its runs directory.
    :type paths: Paths
    :param output: Fresh immutable attempt directory inside the home.
    :type output: Path
    :param cohort_path: Committed exact 280-photograph source-hash document.
    :type cohort_path: Path
    :param schedule_path: Committed CV, readout, and gate document.
    :type schedule_path: Path
    :param membership_path: Archived original split-membership document.
    :type membership_path: Path
    :returns: Training-only two-probe report and predeclared advancement decision.
    :rtype: dict[str, Any]
    :raises ValueError: If protocol/source verification or scientific controls fail.
    """
    parameters = {
        'kind': 'input_access',
        'issue': 58,
        'cohort_sha256': sha256_file(path=cohort_path),
        'schedule_sha256': sha256_file(path=schedule_path),
        'membership_sha256': sha256_file(path=membership_path),
        'representations': list(REPRESENTATIONS),
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        start = perf_counter()
        cohort = _read_json(path=cohort_path)
        schedule = _read_json(path=schedule_path)
        membership = _read_json(path=membership_path)
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        rows = _verify_protocol(
            cfg=cfg,
            paths=paths,
            prepared=prepared,
            cohort=cohort,
            schedule=schedule,
            membership=membership,
            membership_path=membership_path,
            cohort_path=cohort_path,
        )
        features, provenance = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        provenance.update(
            source_config_sha256=config_hash(cfg=cfg),
            source_membership_sha256=cohort['source_membership_sha256'],
            preprocess_cache_key=prepared.key,
            dataset_fingerprint=prepared.fingerprint,
            cohort_sha256=parameters['cohort_sha256'],
            schedule_sha256=parameters['schedule_sha256'],
            validation_test_reserve_scored=False,
        )
        write_json(path=directory / 'provenance.json', value=provenance)
        fit_indices = np.asarray(
            a=[i for i, item in enumerate(cohort['samples']) if item['role'] == 'fit'],
            dtype=np.int64,
        )
        query_indices = np.asarray(
            a=[i for i, item in enumerate(cohort['samples']) if item['role'] == 'query'],
            dtype=np.int64,
        )
        fit_labels = np.asarray(
            a=[cohort['samples'][i]['label'] for i in fit_indices], dtype=np.int64
        )
        models: dict[str, Any] = {}
        fitting: dict[str, Any] = {}
        for name in REPRESENTATIONS:
            model, cv = fit_classifier(
                x_train=features[name][fit_indices],
                y_train=fit_labels,
                pca_components=schedule['readout']['pca_components'],
                c_grid=schedule['readout']['c_grid'],
                cv_folds=schedule['readout']['cv_folds'],
                seed=schedule['readout']['seed'],
                tolerance=schedule['readout']['logistic_tolerance'],
                max_iterations=schedule['readout']['max_iterations'],
            )
            models[name] = model
            fitting[name] = {
                'cv_scores': cv,
                'model': export_classifier(model=model, directory=directory / name / 'model'),
            }
            if (
                perf_counter() - start > schedule['resource_limit']['max_wall_minutes'] * 60
                or psutil.Process().memory_info().rss
                > schedule['resource_limit']['max_rss_gib'] * 1024**3
            ):
                raise RuntimeError('Input-access study exceeded its frozen resource budget.')
        # Query targets are constructed only after both fits and C choices are fixed.
        query_labels = np.asarray(
            a=[cohort['samples'][i]['label'] for i in query_indices], dtype=np.int64
        )
        ids = [item['sample_id'] for item in cohort['samples']]
        scores: dict[str, Any] = {}
        metric_rows: list[dict[str, Any]] = []
        for name in REPRESENTATIONS:
            model = models[name]
            x = features[name]
            target_dir = directory / name
            train_score = score_split(
                model=model,
                x=x[fit_indices],
                targets=fit_labels,
                ids=[ids[i] for i in fit_indices],
                directory=target_dir,
                split='fit',
            )
            query_score = score_split(
                model=model,
                x=x[query_indices],
                targets=query_labels,
                ids=[ids[i] for i in query_indices],
                directory=target_dir,
                split='query',
            )
            scores[name] = {
                'fit': train_score,
                'query': query_score,
                'query_diagnostics': _summarize_queries(
                    model=model, features=x[query_indices], labels=query_labels
                ),
            }
            metric_rows.extend(
                {'representation': name, 'split': split, **metric}
                for split, metric in (('fit', train_score), ('query', query_score))
            )
        write_table(path=directory / 'metrics.parquet', rows=metric_rows)
        passed = all(
            scores[name]['query']['accuracy'] >= 0.25
            and scores[name]['query']['log_loss'] < math.log(20)
            for name in REPRESENTATIONS
        )
        report = {
            'parameters': parameters,
            'provenance': provenance,
            'cohort_counts': {'fit': len(fit_indices), 'query': len(query_indices), 'classes': 20},
            'fitting': fitting,
            'scores': scores,
            'gate': 'advance_to_matched_neural_study' if passed else 'do_not_advance',
            'gate_passed': passed,
            'chance_accuracy': 0.05,
            'query_log_loss_threshold': math.log(20),
            'inference': 'exploratory training-only queries; no validation, test, or reserve score',
        }
        write_json(path=directory / 'report.json', value=report)
    return report

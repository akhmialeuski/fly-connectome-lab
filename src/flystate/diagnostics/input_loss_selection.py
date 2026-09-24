"""Fit-only out-of-fold log-loss selection for transparent image inputs."""

import math
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.artifacts import attempt, export_classifier
from flystate.diagnostics.input_access import (
    REPRESENTATIONS,
    _features,
    _read_json,
    _summarize_queries,
    _verify_protocol,
)
from flystate.diagnostics.probes import score_split
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file
from flystate.readouts.fitting import _classifier, _projection
from flystate.readouts.training import accuracy_metrics
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table


def _resource_guard(start: float, schedule: dict[str, Any]) -> None:
    """Stop on the frozen wall-time or peak resident-memory budget.

    :param start: Perf-counter timestamp at attempt start, in seconds.
    :type start: float
    :param schedule: Frozen wall-minute and peak-GiB limits.
    :type schedule: dict[str, Any]
    :raises RuntimeError: If either resource budget has been exceeded.
    """
    peak = psutil.Process().memory_info().rss
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text(encoding='utf-8').splitlines():
            if line.startswith('VmHWM:'):
                peak = max(peak, int(line.split()[1]) * 1024)
                break
    if (
        perf_counter() - start > schedule['resources']['max_wall_minutes'] * 60
        or peak > schedule['resources']['max_peak_rss_gib'] * 1024**3
    ):
        raise RuntimeError('Input-loss selection exceeded its frozen resource budget.')


def _verify_schedule(
    cfg: ExperimentConfig,
    schedule: dict[str, Any],
    cohort_path: Path,
    parent_schedule_path: Path,
    baseline_path: Path,
) -> None:
    """Check the committed T30 method and exact T29 parent evidence hashes.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param schedule: Committed T30 selection and gate document.
    :type schedule: dict[str, Any]
    :param cohort_path: Original committed T29 cohort file.
    :type cohort_path: Path
    :param parent_schedule_path: Original committed T29 schedule file.
    :type parent_schedule_path: Path
    :param baseline_path: Original completed T29 A1 archive directory.
    :type baseline_path: Path
    :raises ValueError: If source evidence, method, or gate changed.
    """
    expected = {
        'schema_version': 1,
        'issue': 60,
        'source_cohort_sha256': sha256_file(path=cohort_path),
        'source_schedule_sha256': sha256_file(path=parent_schedule_path),
        'source_A1_report_sha256': sha256_file(path=baseline_path / 'report.json'),
        'source_A1_manifest_sha256': sha256_file(path=baseline_path / 'manifest.json'),
        'representations': list(REPRESENTATIONS),
        'fit_roles': ['fit'],
        'query_role': 'query',
        'query_status': 'previously_inspected_development_only',
        'fold_source': 'T29 schedule.cv; five fixed stratified folds',
        'preprocessing': ['StandardScaler', 'PCA'],
        'pca_components': cfg.readout.pca_components,
        'candidate_C': list(cfg.readout.c_grid),
        'seed': cfg.seed,
        'tolerance': 1e-6,
        'max_iterations': 50000,
        'blas_threads': 1,
        'selection_metric': 'pooled_200_row_out_of_fold_natural_log_loss',
        'selection_tie_break': 'smaller_C',
        'out_of_fold_scoring': (
            'sklearn.metrics.log_loss with fixed classes 0..19; '
            'top-1/top-5 from stored probabilities'
        ),
        'gate': {
            'min_oof_top1': 0.25,
            'max_oof_log_loss_exclusive': 'ln(20)',
            'both_representations_required': True,
        },
        'resources': {'max_wall_minutes': 60, 'max_peak_rss_gib': 12},
    }
    if schedule != expected:
        raise ValueError('Input-loss selection differs from the frozen T30 schedule.')


def _cross_validate(
    x: NDArray[np.float32],
    labels: NDArray[np.int64],
    ids: list[str],
    schedule: dict[str, Any],
    parent_folds: list[dict[str, Any]],
    representation: str,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    """Fit every C in every fixed fold and preserve all 200 OOF rows per C.

    :param x: Float32 fit-only features (200,F), original input units.
    :type x: NDArray[np.float32]
    :param labels: Fit-only integer identity labels (200,).
    :type labels: NDArray[np.int64]
    :param ids: Ordered 200 original training sample IDs.
    :type ids: list[str]
    :param schedule: Frozen T30 C grid, PCA, convergence, and resource settings.
    :type schedule: dict[str, Any]
    :param parent_folds: Exact T29 five-fold membership.
    :type parent_folds: list[dict[str, Any]]
    :param representation: Pixels or original encoded current.
    :type representation: str
    :param started: Overall attempt start in perf-counter seconds.
    :type started: float
    :returns: All OOF rows, fold rows, candidate summaries, and selected C.
    :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]
    :raises ValueError: If CV membership, class order, or finite probabilities fail.
    :raises ConvergenceWarning: If any candidate reaches its iteration budget.
    """
    if x.shape[0] != 200 or labels.shape != (200,) or len(ids) != 200:
        raise ValueError('T30 requires exactly 200 fit-only feature rows.')
    classes = np.arange(20, dtype=np.int64)
    candidates = schedule['candidate_C']
    probabilities = {
        value: np.full(shape=(200, 20), fill_value=np.nan, dtype=np.float64) for value in candidates
    }
    fold_by_row = np.full(shape=200, fill_value=-1, dtype=np.int32)
    fold_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=schedule['seed'])
    with threadpool_limits(limits=1, user_api='blas'), warnings.catch_warnings():
        warnings.simplefilter(action='error', category=ConvergenceWarning)
        for fold_index, (training, held_out) in enumerate(
            splitter.split(X=np.arange(len(x)), y=labels)
        ):
            expected = parent_folds[fold_index]
            if (
                expected['index'] != fold_index
                or expected['fit_ids'] != [ids[i] for i in training]
                or expected['held_out_ids'] != [ids[i] for i in held_out]
                or expected['pca_components'] != schedule['pca_components']
            ):
                raise ValueError('OOF fold membership differs from the frozen T29 schedule.')
            projection, fit_features = _projection(
                x=x[training].astype(np.float64),
                components=schedule['pca_components'],
                seed=schedule['seed'],
            )
            query_features = projection.transform(X=x[held_out].astype(np.float64))
            fold_by_row[held_out] = fold_index
            for value in candidates:
                classifier = _classifier(
                    c_value=value,
                    seed=schedule['seed'],
                    tolerance=schedule['tolerance'],
                    max_iterations=schedule['max_iterations'],
                )
                classifier.fit(X=fit_features, y=labels[training])
                if not np.array_equal(classifier.classes_, classes):
                    raise ValueError('OOF classifier class order differs from the frozen cohort.')
                prediction = classifier.predict_proba(X=query_features)
                probabilities[value][held_out] = prediction
                fold_rows.append(
                    {
                        'representation': representation,
                        'C': value,
                        'fold': fold_index,
                        'fit_rows': len(training),
                        'held_out_rows': len(held_out),
                        'iterations': int(classifier.n_iter_[0]),
                        'accuracy': float(
                            np.mean(a=classes[np.argmax(prediction, axis=1)] == labels[held_out])
                        ),
                        'log_loss': float(
                            log_loss(y_true=labels[held_out], y_proba=prediction, labels=classes)
                        ),
                    }
                )
                _resource_guard(start=started, schedule=schedule)
    if np.any(fold_by_row < 0):
        raise ValueError('OOF partition omitted a fit photograph.')
    oof_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for value in candidates:
        predicted = probabilities[value]
        if not np.isfinite(predicted).all() or not np.allclose(
            a=predicted.sum(axis=1), b=1, atol=1e-12, rtol=0
        ):
            raise ValueError('OOF probability matrix is incomplete or invalid.')
        metrics = accuracy_metrics(probabilities=predicted, targets=labels, classes=classes)
        summaries.append(
            {
                'representation': representation,
                'C': value,
                **metrics,
                'log_loss': float(log_loss(y_true=labels, y_proba=predicted, labels=classes)),
            }
        )
        oof_rows.extend(
            {
                'representation': representation,
                'C': value,
                'sample_id': sample_id,
                'fold': int(fold_by_row[index]),
                'y_true': int(labels[index]),
                'y_pred': int(classes[np.argmax(probability)]),
                'probabilities': probability.tolist(),
            }
            for index, (sample_id, probability) in enumerate(zip(ids, predicted, strict=True))
        )
    selected = min(summaries, key=lambda row: (row['log_loss'], row['C']))['C']
    return oof_rows, fold_rows, summaries, float(selected)


def run_input_loss_selection(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    cohort_path: Path,
    parent_schedule_path: Path,
    membership_path: Path,
    baseline_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    """Choose input-probe C by fit-only OOF log loss and archive all candidates.

    :param cfg: Original 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home; attempts stay beneath its runs directory.
    :type paths: Paths
    :param output: New immutable T30 attempt directory.
    :type output: Path
    :param cohort_path: T29 committed 280-image cohort document.
    :type cohort_path: Path
    :param parent_schedule_path: T29 committed exact CV-fold document.
    :type parent_schedule_path: Path
    :param membership_path: Archived original split membership.
    :type membership_path: Path
    :param baseline_path: Archived completed T29 A1 attempt.
    :type baseline_path: Path
    :param schedule_path: Committed T30 scoring and gate document.
    :type schedule_path: Path
    :returns: Complete OOF gate and descriptive prior-query results.
    :rtype: dict[str, Any]
    :raises ValueError: If source, folds, or protocol differ.
    """
    parameters = {
        'kind': 'input_loss_selection',
        'issue': 60,
        'schedule_sha256': sha256_file(path=schedule_path),
        'cohort_sha256': sha256_file(path=cohort_path),
        'parent_schedule_sha256': sha256_file(path=parent_schedule_path),
        'baseline_report_sha256': sha256_file(path=baseline_path / 'report.json'),
        'representations': list(REPRESENTATIONS),
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        started = perf_counter()
        cohort = _read_json(path=cohort_path)
        parent_schedule = _read_json(path=parent_schedule_path)
        membership = _read_json(path=membership_path)
        schedule = _read_json(path=schedule_path)
        baseline = _read_json(path=baseline_path / 'report.json')
        _verify_schedule(
            cfg=cfg,
            schedule=schedule,
            cohort_path=cohort_path,
            parent_schedule_path=parent_schedule_path,
            baseline_path=baseline_path,
        )
        if (
            baseline['parameters']['kind'] != 'input_access'
            or baseline['parameters']['cohort_sha256'] != parameters['cohort_sha256']
            or baseline['parameters']['schedule_sha256'] != parameters['parent_schedule_sha256']
            or baseline['gate_passed'] is not False
        ):
            raise ValueError('T29 baseline differs from the frozen source study.')
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        rows = _verify_protocol(
            cfg=cfg,
            paths=paths,
            prepared=prepared,
            cohort=cohort,
            schedule=parent_schedule,
            membership=membership,
            membership_path=membership_path,
            cohort_path=cohort_path,
        )
        features, provenance = _features(cfg=cfg, paths=paths, prepared=prepared, rows=rows)
        if (
            provenance['feature_sha256'] != baseline['provenance']['feature_sha256']
            or provenance['brain_files_sha256'] != baseline['provenance']['brain_files_sha256']
            or provenance['encoder'] != baseline['provenance']['encoder']
        ):
            raise ValueError('T30 input features differ from completed T29 baseline.')
        provenance.update(
            source_config_sha256=config_hash(cfg=cfg),
            source_cohort_sha256=parameters['cohort_sha256'],
            source_parent_schedule_sha256=parameters['parent_schedule_sha256'],
            source_baseline_report_sha256=parameters['baseline_report_sha256'],
            dataset_fingerprint=prepared.fingerprint,
            preprocess_cache_key=prepared.key,
            validation_test_reserve_scored=False,
            prior_queries_already_inspected=True,
        )
        write_json(path=directory / 'provenance.json', value=provenance)
        _resource_guard(start=started, schedule=schedule)
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
        fit_ids = [cohort['samples'][i]['sample_id'] for i in fit_indices]
        all_oof: list[dict[str, Any]] = []
        all_folds: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        selected: dict[str, float] = {}
        for name in REPRESENTATIONS:
            oof, fold_rows, candidates, chosen = _cross_validate(
                x=features[name][fit_indices],
                labels=fit_labels,
                ids=fit_ids,
                schedule=schedule,
                parent_folds=parent_schedule['cv'],
                representation=name,
                started=started,
            )
            all_oof.extend(oof)
            all_folds.extend(fold_rows)
            all_candidates.extend(candidates)
            selected[name] = chosen
        write_table(path=directory / 'oof-predictions.parquet', rows=all_oof)
        write_table(path=directory / 'fold-metrics.parquet', rows=all_folds)
        write_table(path=directory / 'candidate-metrics.parquet', rows=all_candidates)
        chosen_rows = {
            name: next(
                row
                for row in all_candidates
                if row['representation'] == name and row['C'] == selected[name]
            )
            for name in REPRESENTATIONS
        }
        gate_passed = all(
            chosen_rows[name]['accuracy'] >= schedule['gate']['min_oof_top1']
            and chosen_rows[name]['log_loss'] < math.log(20)
            for name in REPRESENTATIONS
        )
        # The OOF selection and gate are fixed before prior query labels are constructed.
        models: dict[str, Pipeline] = {}
        model_metadata: dict[str, Any] = {}
        with threadpool_limits(limits=1, user_api='blas'), warnings.catch_warnings():
            warnings.simplefilter(action='error', category=ConvergenceWarning)
            for name in REPRESENTATIONS:
                projection, transformed = _projection(
                    x=features[name][fit_indices].astype(np.float64),
                    components=schedule['pca_components'],
                    seed=schedule['seed'],
                )
                classifier = _classifier(
                    c_value=selected[name],
                    seed=schedule['seed'],
                    tolerance=schedule['tolerance'],
                    max_iterations=schedule['max_iterations'],
                )
                classifier.fit(X=transformed, y=fit_labels)
                model = Pipeline(steps=[*projection.steps, ('classifier', classifier)])
                models[name] = model
                model_metadata[name] = export_classifier(
                    model=model, directory=directory / name / 'model'
                )
                _resource_guard(start=started, schedule=schedule)
        query_labels = np.asarray(
            a=[cohort['samples'][i]['label'] for i in query_indices], dtype=np.int64
        )
        ids = [item['sample_id'] for item in cohort['samples']]
        scores: dict[str, Any] = {}
        metric_rows: list[dict[str, Any]] = []
        for name in REPRESENTATIONS:
            fit_score = score_split(
                model=models[name],
                x=features[name][fit_indices],
                targets=fit_labels,
                ids=[ids[i] for i in fit_indices],
                directory=directory / name,
                split='fit',
            )
            query_score = score_split(
                model=models[name],
                x=features[name][query_indices],
                targets=query_labels,
                ids=[ids[i] for i in query_indices],
                directory=directory / name,
                split='query',
            )
            scores[name] = {
                'fit': fit_score,
                'query_descriptive': query_score,
                'query_diagnostics': _summarize_queries(
                    model=models[name], features=features[name][query_indices], labels=query_labels
                ),
                'T29_query_baseline': baseline['scores'][name]['query'],
            }
            metric_rows.extend(
                {'representation': name, 'split': split, **score}
                for split, score in (('fit', fit_score), ('query_descriptive', query_score))
            )
        write_table(path=directory / 'metrics.parquet', rows=metric_rows)
        report = {
            'parameters': parameters,
            'provenance': provenance,
            'selected_C': selected,
            'selected_oof': chosen_rows,
            'all_candidates': all_candidates,
            'model': model_metadata,
            'scores': scores,
            'gate_passed': gate_passed,
            'gate': 'advance_to_matched_neural_study' if gate_passed else 'do_not_advance',
            'chance_accuracy': 0.05,
            'log_loss_threshold': math.log(20),
            'inference': (
                'fit-only exploratory OOF gate; prior queries are descriptive development only'
            ),
        }
        write_json(path=directory / 'report.json', value=report)
    return report

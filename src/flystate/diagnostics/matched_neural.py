"""Matched fit-only identity readouts from immutable original MaleCNS traces."""

import json
import math
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from flystate.diagnostics.artifacts import attempt, export_classifier
from flystate.diagnostics.input_access import _read_json, _summarize_queries
from flystate.diagnostics.input_loss_selection import _cross_validate, _resource_guard
from flystate.diagnostics.noise import load_noise_trace
from flystate.diagnostics.probes import score_split
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_file, stable_int
from flystate.readouts.fitting import _classifier, _projection
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

CASES: tuple[str, ...] = ('N0', 'N1', 'OFF')
INPUT_REFERENCE: str = (
    'T30 encoded_current selected C=0.01 OOF probability rows for the same 200 fit sample IDs'
)


def _verify_schedule(
    cfg: ExperimentConfig,
    schedule: dict[str, Any],
    cohort_path: Path,
    parent_schedule_path: Path,
    t30_schedule_path: Path,
    t30_path: Path,
    membership_path: Path,
) -> None:
    """Require the precommitted source hashes, exact readout, and comparison gate.

    :param cfg: Original persistent fly experiment configuration.
    :type cfg: ExperimentConfig
    :param schedule: Frozen T31 source and analysis schedule.
    :type schedule: dict[str, Any]
    :param cohort_path: Exact committed T29 280-photo cohort file.
    :type cohort_path: Path
    :param parent_schedule_path: Exact committed T29 fold schedule.
    :type parent_schedule_path: Path
    :param t30_schedule_path: Exact committed T30 input schedule.
    :type t30_schedule_path: Path
    :param t30_path: Archived completed T30 attempt.
    :type t30_path: Path
    :param membership_path: Archived original membership file.
    :type membership_path: Path
    :raises ValueError: If source or scientific decisions changed.
    """
    if (
        schedule['schema_version'] != 1
        or schedule['issue'] != 62
        or schedule['source_config'] != 'configs/celeba-smoke.yaml'
        or schedule['source_cohort_sha256'] != sha256_file(path=cohort_path)
        or schedule['source_parent_schedule_sha256'] != sha256_file(path=parent_schedule_path)
        or schedule['source_t30_schedule_sha256'] != sha256_file(path=t30_schedule_path)
        or schedule['source_t30_report_sha256'] != sha256_file(path=t30_path / 'report.json')
        or schedule['source_t30_oof_sha256']
        != sha256_file(path=t30_path / 'oof-predictions.parquet')
        or schedule['source_membership_sha256'] != sha256_file(path=membership_path)
        or schedule['case_order'] != list(CASES)
        or set(schedule['traces']) != set(CASES)
        or schedule['trace_precision'] != 'float32'
        or schedule['history'] != 'all'
        or schedule['neural_blocks'] != list(cfg.readout.features)
        or schedule['neural_features_per_observation'] != 2628
        or schedule['observation_count'] != cfg.episodes.steps
        or schedule['feature_dimension'] != 42048
        or schedule['input_reference'] != INPUT_REFERENCE
        or schedule['candidate_C'] != list(cfg.readout.c_grid)
        or schedule['pca_components'] != cfg.readout.pca_components
        or schedule['cv_folds'] != cfg.readout.cv_folds
        or schedule['seed'] != cfg.seed
        or schedule['tolerance'] != 1e-6
        or schedule['max_iterations'] != 50000
        or schedule['blas_threads'] != 1
        or schedule['selection_metric'] != 'pooled_200_row_out_of_fold_natural_log_loss'
        or schedule['selection_tie_break'] != 'smaller_C'
        or schedule['cluster_bootstrap']
        != {
            'samples': 10000,
            'seed': 0,
            'namespace': 't31-identity-bootstrap',
            'cluster_count': 20,
            'percentiles': [2.5, 97.5],
            'percentile_method': 'linear',
        }
        or schedule['gate']
        != {
            'required_cases': ['N0', 'N1'],
            'max_neural_minus_input_accuracy_pp': -10.0,
            'bootstrap_upper_pp_exclusive': 0.0,
            'require_neural_log_loss_above_input': True,
        }
        or schedule['query_status'] != 'previously_inspected_development_only'
        or schedule['resources'] != {'max_wall_minutes': 60, 'max_peak_rss_gib': 12}
        or cfg.memory.mode != 'persistent'
        or cfg.model_dump(mode='json')
        != json.loads((t30_path / 'config.json').read_text(encoding='utf-8'))
    ):
        raise ValueError('Matched-neural analysis differs from the frozen T31 schedule.')
    expected_traces = {
        'N0': ('seed0', 0, True),
        'N1': ('seed1', 1, True),
        'OFF': ('off', 0, False),
    }
    for case, (name, seed, enabled) in expected_traces.items():
        trace = schedule['traces'][case]
        archived = (
            f'research/sequential-visual-memory/2026-09-23-noise-recognition/snapshot/traces/{name}'
        )
        if (
            trace['working_relative_path']
            != f'runs/diagnostics/2026-09-23-noise-recognition/traces/{name}'
            or trace['archived_relative_path'] != archived
            or trace['episode_seed'] != seed
            or trace['episode_noise_enabled'] is not enabled
            or len(trace['manifest_sha256']) != 64
            or len(trace['features_sha256']) != 64
            or len(trace['samples_sha256']) != 64
        ):
            raise ValueError(f'Case {case} changed its frozen trace identity.')


def _reference_oof(
    t30_path: Path, t30_report: dict[str, Any], fit_ids: list[str], labels: NDArray[np.int64]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the fixed T30 encoded-current OOF control on the same 200 fit IDs.

    :param t30_path: Completed archived T30 attempt.
    :type t30_path: Path
    :param t30_report: Verified T30 report and selected C.
    :type t30_report: dict[str, Any]
    :param fit_ids: Exact ordered 200 T29 fit sample IDs.
    :type fit_ids: list[str]
    :param labels: Exact ordered identity labels (200,), int64.
    :type labels: NDArray[np.int64]
    :returns: Ordered fixed input OOF records and selected candidate metrics.
    :rtype: tuple[list[dict[str, Any]], dict[str, Any]]
    :raises ValueError: If parent gate, C, membership, or prediction rows differ.
    """
    if t30_report['gate_passed'] is not True or t30_report['selected_C'] != {
        'pixels': 0.01,
        'encoded_current': 0.01,
    }:
        raise ValueError('T30 input reference must be the completed passing C=0.01 control.')
    records = pq.read_table(source=t30_path / 'oof-predictions.parquet').to_pylist()
    selected = [
        row for row in records if row['representation'] == 'encoded_current' and row['C'] == 0.01
    ]
    if (
        len(selected) != 200
        or [row['sample_id'] for row in selected] != fit_ids
        or [row['y_true'] for row in selected] != labels.tolist()
        or len({row['sample_id'] for row in selected}) != 200
        or any(len(row['probabilities']) != 20 for row in selected)
    ):
        raise ValueError('T30 OOF control differs from the exact T31 fit cohort.')
    expected = t30_report['selected_oof']['encoded_current']
    correct = np.asarray(a=[row['y_pred'] == row['y_true'] for row in selected], dtype=bool)
    if int(np.count_nonzero(a=correct)) != round(expected['accuracy'] * 200) or expected[
        'log_loss'
    ] >= math.log(20):
        raise ValueError('T30 OOF control differs from its completed report.')
    return selected, expected


def _paired_analysis(
    case: str,
    input_rows: list[dict[str, Any]],
    neural_rows: list[dict[str, Any]],
    input_metrics: dict[str, Any],
    neural_metrics: dict[str, Any],
    bootstrap_draws: NDArray[np.int64],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute paired identity-cluster accuracy intervals and error overlap.

    :param case: N0, N1, or OFF neural condition.
    :type case: str
    :param input_rows: Ordered fixed input OOF records, length 200.
    :type input_rows: list[dict[str, Any]]
    :param neural_rows: Ordered selected-C neural OOF records, length 200.
    :type neural_rows: list[dict[str, Any]]
    :param input_metrics: T30 selected-C OOF metrics.
    :type input_metrics: dict[str, Any]
    :param neural_metrics: Selected neural OOF metrics.
    :type neural_metrics: dict[str, Any]
    :param bootstrap_draws: Seeded identity-cluster indices (10000,20), int64.
    :type bootstrap_draws: NDArray[np.int64]
    :returns: Comparison summary, 200 paired image rows, and 10000 bootstrap rows.
    :rtype: tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]
    :raises ValueError: If query alignment, class counts, or bootstrap shape differ.
    """
    if (
        len(input_rows) != 200
        or len(neural_rows) != 200
        or bootstrap_draws.shape != (10000, 20)
        or bootstrap_draws.dtype.kind not in 'iu'
        or np.any(bootstrap_draws < 0)
        or np.any(bootstrap_draws >= 20)
    ):
        raise ValueError('Matched neural comparison requires 200 aligned rows and 20 clusters.')
    paired: list[dict[str, Any]] = []
    differences = np.empty(shape=200, dtype=np.float64)
    labels = np.empty(shape=200, dtype=np.int64)
    overlap = {'both_correct': 0, 'input_only': 0, 'neural_only': 0, 'both_wrong': 0}
    for index, (input_row, neural_row) in enumerate(zip(input_rows, neural_rows, strict=True)):
        if (
            input_row['sample_id'] != neural_row['sample_id']
            or input_row['y_true'] != neural_row['y_true']
        ):
            raise ValueError('Input and neural OOF predictions are not paired by image.')
        label = int(input_row['y_true'])
        input_correct = bool(input_row['y_pred'] == label)
        neural_correct = bool(neural_row['y_pred'] == label)
        labels[index] = label
        differences[index] = float(neural_correct) - float(input_correct)
        category = (
            'both_correct'
            if input_correct and neural_correct
            else 'input_only'
            if input_correct
            else 'neural_only'
            if neural_correct
            else 'both_wrong'
        )
        overlap[category] += 1
        paired.append(
            {
                'case': case,
                'sample_id': input_row['sample_id'],
                'label': label,
                'input_pred': input_row['y_pred'],
                'neural_pred': neural_row['y_pred'],
                'input_correct': input_correct,
                'neural_correct': neural_correct,
                'category': category,
            }
        )
    if set(labels.tolist()) != set(range(20)) or any(
        np.count_nonzero(labels == i) != 10 for i in range(20)
    ):
        raise ValueError('Matched comparison requires ten fit photographs for every identity.')
    per_identity = np.asarray(
        a=[np.mean(a=differences[labels == label]) for label in range(20)], dtype=np.float64
    )
    distribution = np.mean(a=per_identity[bootstrap_draws], axis=1) * 100
    low, high = np.percentile(a=distribution, q=[2.5, 97.5], method='linear')
    observed = float(np.mean(a=differences) * 100)
    if not math.isclose(
        observed, 100 * (neural_metrics['accuracy'] - input_metrics['accuracy']), abs_tol=1e-10
    ):
        raise ValueError('Matched accuracy gap differs from stored OOF metrics.')
    summary = {
        'case': case,
        'input_accuracy': input_metrics['accuracy'],
        'neural_accuracy': neural_metrics['accuracy'],
        'neural_minus_input_accuracy_pp': observed,
        'input_log_loss': input_metrics['log_loss'],
        'neural_log_loss': neural_metrics['log_loss'],
        'cluster_interval_low_pp': float(low),
        'cluster_interval_high_pp': float(high),
        'bootstrap_samples': len(distribution),
        'error_overlap': overlap,
    }
    distribution_rows = [
        {'case': case, 'draw': i, 'neural_minus_input_accuracy_pp': float(value)}
        for i, value in enumerate(distribution)
    ]
    return summary, paired, distribution_rows


def run_matched_neural(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    cohort_path: Path,
    parent_schedule_path: Path,
    t30_schedule_path: Path,
    t30_path: Path,
    membership_path: Path,
    schedule_path: Path,
) -> dict[str, Any]:
    """Fit fixed neural traces against identical input OOF identity measurements.

    :param cfg: Original persistent 20-identity experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home containing three immutable source traces.
    :type paths: Paths
    :param output: Fresh immutable diagnostic attempt directory in that home.
    :type output: Path
    :param cohort_path: T29 280-photo source-hash cohort file.
    :type cohort_path: Path
    :param parent_schedule_path: T29 exact CV-fold schedule.
    :type parent_schedule_path: Path
    :param t30_schedule_path: T30 frozen input-selection schedule.
    :type t30_schedule_path: Path
    :param t30_path: Archived completed T30 input-reference attempt.
    :type t30_path: Path
    :param membership_path: Archived original dataset-split membership.
    :type membership_path: Path
    :param schedule_path: Committed T31 trace/comparison/gate schedule.
    :type schedule_path: Path
    :returns: Complete candidate results, paired OOF gate, and descriptive queries.
    :rtype: dict[str, Any]
    :raises ValueError: If any source, fold, probability, or geometry control fails.
    """
    parameters = {
        'kind': 'matched_neural_access',
        'issue': 62,
        'schedule_sha256': sha256_file(path=schedule_path),
        'cohort_sha256': sha256_file(path=cohort_path),
        'input_oof_sha256': sha256_file(path=t30_path / 'oof-predictions.parquet'),
        'case_order': list(CASES),
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        started = perf_counter()
        schedule = _read_json(path=schedule_path)
        cohort = _read_json(path=cohort_path)
        parent_schedule = _read_json(path=parent_schedule_path)
        t30_report = _read_json(path=t30_path / 'report.json')
        _verify_schedule(
            cfg=cfg,
            schedule=schedule,
            cohort_path=cohort_path,
            parent_schedule_path=parent_schedule_path,
            t30_schedule_path=t30_schedule_path,
            t30_path=t30_path,
            membership_path=membership_path,
        )
        if parameters['input_oof_sha256'] != schedule['source_t30_oof_sha256']:
            raise ValueError('T30 input OOF file differs from the frozen T31 reference.')
        fit = [item for item in cohort['samples'] if item['role'] == 'fit']
        query = [item for item in cohort['samples'] if item['role'] == 'query']
        fit_ids = [item['sample_id'] for item in fit]
        fit_labels = np.asarray(a=[item['label'] for item in fit], dtype=np.int64)
        query_ids = [item['sample_id'] for item in query]
        if len(fit) != 200 or len(query) != 80:
            raise ValueError('T31 cohort no longer has frozen 200/80 roles.')
        reference, reference_metrics = _reference_oof(
            t30_path=t30_path, t30_report=t30_report, fit_ids=fit_ids, labels=fit_labels
        )
        brain_hashes = {
            name: sha256_file(path=paths.brain / name) for name in ('brain.npz', 'weights.npz')
        }
        if brain_hashes != t30_report['provenance']['brain_files_sha256']:
            raise ValueError('Current graph files differ from the frozen input reference.')
        _resource_guard(start=started, schedule=schedule)
        generator = np.random.default_rng(
            seed=np.random.SeedSequence(
                entropy=[
                    schedule['cluster_bootstrap']['seed'],
                    stable_int(key=schedule['cluster_bootstrap']['namespace']),
                ]
            )
        )
        draws = generator.integers(low=0, high=20, size=(10000, 20), dtype=np.int64)
        all_oof: list[dict[str, Any]] = []
        all_folds: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        all_paired: list[dict[str, Any]] = []
        all_bootstrap: list[dict[str, Any]] = []
        comparisons: dict[str, Any] = {}
        selected: dict[str, float] = {}
        feature_sources: dict[str, Any] = {}
        case_features: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        for case in CASES:
            trace_spec = schedule['traces'][case]
            source = paths.home / trace_spec['working_relative_path']
            if (
                sha256_file(path=source / 'manifest.json') != trace_spec['manifest_sha256']
                or sha256_file(path=source / 'features.npz') != trace_spec['features_sha256']
                or sha256_file(path=source / 'samples.parquet') != trace_spec['samples_sha256']
            ):
                raise ValueError(f'Case {case} differs from its frozen native trace files.')
            data = load_noise_trace(
                cfg=cfg,
                paths=paths,
                source=source,
                precision='float32',
                history='all',
                features='both',
            )
            trace_parameters = data.provenance['parameters']
            trace_fields = data.provenance['cache_key_fields']
            if (
                data.x.shape != (340, 42048)
                or data.x.dtype != np.float32
                or data.provenance['source_config_sha256'] != config_hash(cfg=cfg)
                or data.provenance['dataset_fingerprint']
                != t30_report['provenance']['dataset_fingerprint']
                or data.provenance['membership_sha256'] != cohort['source_membership_sha256']
                or trace_fields['brain_files_sha256'] != brain_hashes
                or trace_fields['encoder'] != cfg.encoder.model_dump(mode='json')
                or data.provenance['readout_neurons'] != 1314
                or data.provenance['feature_blocks'] != list(cfg.readout.features)
                or trace_parameters['episode_seed'] != trace_spec['episode_seed']
                or trace_parameters['episode_noise_enabled']
                is not trace_spec['episode_noise_enabled']
                or data.provenance['source_manifest_sha256'] != trace_spec['manifest_sha256']
                or data.provenance['features_sha256'] != trace_spec['features_sha256']
            ):
                raise ValueError(f'Case {case} trace geometry or noise treatment changed.')
            positions = {sample.sample_id: i for i, sample in enumerate(data.samples)}
            if (
                len(positions) != 340
                or {sample.sample_id for sample in data.samples if sample.split == 'train'}
                != set(item['sample_id'] for item in cohort['samples'])
                or any(
                    data.samples[positions[item['sample_id']]].label != item['label']
                    or data.samples[positions[item['sample_id']]].split != 'train'
                    for item in cohort['samples']
                )
            ):
                raise ValueError(f'Case {case} training membership or labels changed.')
            x_fit = data.x[[positions[sample_id] for sample_id in fit_ids]]
            x_query = data.x[[positions[sample_id] for sample_id in query_ids]]
            case_features[case] = (x_fit, x_query)
            feature_sources[case] = {
                'source_trace': trace_spec['working_relative_path'],
                'source_manifest_sha256': trace_spec['manifest_sha256'],
                'features_sha256': trace_spec['features_sha256'],
                'selected_fit_shape': list(x_fit.shape),
                'selected_query_shape': list(x_query.shape),
                'validation_test_reserve_used_for_fit_or_scoring': False,
            }
            oof, folds, candidates, chosen = _cross_validate(
                x=x_fit,
                labels=fit_labels,
                ids=fit_ids,
                schedule=schedule,
                parent_folds=parent_schedule['cv'],
                representation=case,
                started=started,
            )
            all_oof.extend(oof)
            all_folds.extend(folds)
            all_candidates.extend(candidates)
            selected[case] = chosen
            chosen_rows = [row for row in oof if row['C'] == chosen]
            chosen_metrics = next(row for row in candidates if row['C'] == chosen)
            comparison, paired, bootstrap = _paired_analysis(
                case=case,
                input_rows=reference,
                neural_rows=chosen_rows,
                input_metrics=reference_metrics,
                neural_metrics=chosen_metrics,
                bootstrap_draws=draws,
            )
            comparisons[case] = comparison
            all_paired.extend(paired)
            all_bootstrap.extend(bootstrap)
            _resource_guard(start=started, schedule=schedule)
        gate_passed = all(
            comparisons[case]['neural_minus_input_accuracy_pp']
            <= schedule['gate']['max_neural_minus_input_accuracy_pp']
            and comparisons[case]['cluster_interval_high_pp']
            < schedule['gate']['bootstrap_upper_pp_exclusive']
            and comparisons[case]['neural_log_loss'] > comparisons[case]['input_log_loss']
            for case in schedule['gate']['required_cases']
        )
        write_table(path=directory / 'oof-predictions.parquet', rows=all_oof)
        write_table(path=directory / 'fold-metrics.parquet', rows=all_folds)
        write_table(path=directory / 'candidate-metrics.parquet', rows=all_candidates)
        write_table(path=directory / 'paired-predictions.parquet', rows=all_paired)
        write_table(path=directory / 'bootstrap-differences.parquet', rows=all_bootstrap)
        write_table(path=directory / 'comparison-metrics.parquet', rows=list(comparisons.values()))
        write_json(path=directory / 'source-provenance.json', value=feature_sources)
        # Query labels are used only after all three C choices and the OOF gate are fixed.
        query_labels = np.asarray(a=[item['label'] for item in query], dtype=np.int64)
        scores: dict[str, Any] = {}
        model_metadata: dict[str, Any] = {}
        metric_rows: list[dict[str, Any]] = []
        with threadpool_limits(limits=1, user_api='blas'), warnings.catch_warnings():
            warnings.simplefilter(action='error', category=ConvergenceWarning)
            for case in CASES:
                x_fit, x_query = case_features[case]
                projection, transformed = _projection(
                    x=x_fit.astype(np.float64),
                    components=schedule['pca_components'],
                    seed=schedule['seed'],
                )
                classifier = _classifier(
                    c_value=selected[case],
                    seed=schedule['seed'],
                    tolerance=schedule['tolerance'],
                    max_iterations=schedule['max_iterations'],
                )
                classifier.fit(X=transformed, y=fit_labels)
                model = Pipeline(steps=[*projection.steps, ('classifier', classifier)])
                model_metadata[case] = export_classifier(
                    model=model, directory=directory / case / 'model'
                )
                fit_score = score_split(
                    model=model,
                    x=x_fit,
                    targets=fit_labels,
                    ids=fit_ids,
                    directory=directory / case,
                    split='fit',
                )
                query_score = score_split(
                    model=model,
                    x=x_query,
                    targets=query_labels,
                    ids=query_ids,
                    directory=directory / case,
                    split='query',
                )
                scores[case] = {
                    'fit': fit_score,
                    'query_descriptive': query_score,
                    'query_diagnostics': _summarize_queries(
                        model=model, features=x_query, labels=query_labels
                    ),
                }
                metric_rows.extend(
                    {'case': case, 'split': split, **metrics}
                    for split, metrics in (('fit', fit_score), ('query_descriptive', query_score))
                )
                _resource_guard(start=started, schedule=schedule)
        write_table(path=directory / 'metrics.parquet', rows=metric_rows)
        report = {
            'parameters': parameters,
            'source_provenance': feature_sources,
            'input_reference': reference_metrics,
            'selected_C': selected,
            'all_candidates': all_candidates,
            'comparisons': comparisons,
            'model': model_metadata,
            'scores': scores,
            'gate_passed': gate_passed,
            'gate': 'sampled_state_loss_supported'
            if gate_passed
            else 'matched_comparison_inconclusive',
            'inference': (
                'exploratory fit-only matched comparison; reused queries are descriptive; '
                'no equation-level or sequential-memory claim'
            ),
        }
        write_json(path=directory / 'report.json', value=report)
    return report

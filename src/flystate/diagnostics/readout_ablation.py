"""Verify and analyze a preregistered fixed-trace neural readout screen."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, softmax

from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.diagnostics.data import ProbeData
from flystate.diagnostics.noise import load_noise_trace
from flystate.diagnostics.noise_analysis import identity_cluster_interval
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table

EXPECTED_CASES: tuple[tuple[str, str, str, int | None], ...] = (
    ('B0', 'both', 'last', 60),
    ('B1', 'voltage', 'last', 60),
    ('B2', 'spike_trace', 'last', 60),
    ('B3', 'both', 'last', 20),
    ('B4', 'both', 'last', 120),
    ('B5', 'both', 'last', 240),
    ('B6', 'both', 'last', None),
    ('B7', 'both', 'all', 60),
)
CASE_IDS: tuple[str, ...] = tuple(case[0] for case in EXPECTED_CASES)
MAX_ITERATIONS: int = 50000
TRAIN_PER_CLASS: int = 14
BOOTSTRAP_RESAMPLES: int = 2000
BOOTSTRAP_NAMESPACE: str = 'readout-ablation-identity-bootstrap'
MINIMUM_IMPROVEMENT_PP: float = 10.0
PROBABILITY_TOLERANCE: float = 1e-12
HISTORY_RSS_LIMIT_BYTES: int = 12 * 1024**3
HISTORY_WALL_LIMIT_SECONDS: int = 1800


def replay_exported_predictions(
    data: ProbeData, directory: Path, expected_sha256: str
) -> dict[str, float | int]:
    """Recalculate saved probabilities solely from exported numeric model arrays.

    :param data: Float32 neural features (N,F) and matching development sample IDs.
    :type data: ProbeData
    :param directory: Completed classifier attempt with train and validation Parquet files.
    :type directory: Path
    :param expected_sha256: Recorded SHA-256 of the fitted NPZ coefficient file.
    :type expected_sha256: str
    :returns: Number of independently replayed rows and maximum probability error.
    :rtype: dict[str, float | int]
    :raises ValueError: If the saved coefficients, sample IDs, labels or predictions disagree.
    """
    weights = directory / 'model' / 'weights.npz'
    if sha256_file(path=weights) != expected_sha256:
        raise ValueError('Classifier coefficient hash does not match its report.')
    with np.load(file=weights, allow_pickle=False) as arrays:
        x = data.x.astype(np.float64)
        transformed = (x - arrays['scaler_mean']) / arrays['scaler_scale']
        if 'pca_components' in arrays.files:
            transformed = (transformed - arrays['pca_mean']) @ arrays['pca_components'].T
        logits = transformed @ arrays['coef'].T + arrays['intercept']
        classes = arrays['classes']
        probabilities = (
            np.column_stack((1 - expit(logits[:, 0]), expit(logits[:, 0])))
            if len(classes) == 2
            else softmax(logits, axis=1)
        )
    by_id = {sample.sample_id: (index, sample) for index, sample in enumerate(data.samples)}
    if len(by_id) != len(data.samples):
        raise ValueError('Native trace has duplicate sample IDs.')
    seen: set[str] = set()
    maximum_error = 0.0
    for split, filename in (
        ('train', 'train-predictions.parquet'),
        ('val', 'validation-predictions.parquet'),
    ):
        rows = read_table(path=directory / filename)
        for row in rows:
            sample_id = row['sample_id']
            if sample_id in seen or sample_id not in by_id:
                raise ValueError(
                    'Saved prediction sample IDs are duplicated or absent from the trace.'
                )
            seen.add(sample_id)
            index, sample = by_id[sample_id]
            expected = probabilities[index]
            saved = np.asarray(a=row['probabilities'], dtype=np.float64)
            if (
                sample.split != split
                or sample.label != row['y_true']
                or saved.shape != expected.shape
                or row['y_pred'] != int(classes[np.argmax(a=expected)])
                or not np.isfinite(saved).all()
            ):
                raise ValueError('Saved prediction does not match the native trace sample.')
            error = float(np.max(a=np.abs(saved - expected)))
            maximum_error = max(maximum_error, error)
            if error > PROBABILITY_TOLERANCE:
                raise ValueError('Exported classifier arrays do not replay saved probabilities.')
    if seen != set(by_id):
        raise ValueError('Not every native train/validation sample has one saved prediction.')
    return {'prediction_rows': len(seen), 'max_probability_error': maximum_error}


def _read_case(
    cfg: ExperimentConfig,
    paths: Paths,
    source: Path,
    directory: Path,
    specification: dict[str, Any],
    membership_sha256: str,
) -> dict[str, Any]:
    """Verify one declared attempt, then replay its model and split predictions.

    :param cfg: Frozen source configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param source: Verified seed-zero float32 native trace.
    :type source: Path
    :param directory: Immutable readout attempt.
    :type directory: Path
    :param specification: One frozen schedule case.
    :type specification: dict[str, Any]
    :param membership_sha256: Frozen train/validation membership digest.
    :type membership_sha256: str
    :returns: Attempt status, fit evidence and aligned predictions when available.
    :rtype: dict[str, Any]
    :raises ValueError: If parameters, provenance, integrity or predictions differ.
    """
    manifest = verify_attempt_inventory(directory=directory, paths=paths)
    parameters = manifest['parameters']
    expected = {
        'representation': 'neural',
        'history': specification['history'],
        'features': specification['features'],
        'pca_components': specification['components'],
        'label_mode': 'true',
        'max_iterations': MAX_ITERATIONS,
        'train_per_class': TRAIN_PER_CLASS,
        'subset_seed': 0,
        'trace_precision': 'float32',
    }
    if (
        any(parameters.get(key) != value for key, value in expected.items())
        or parameters.get('kind') != 'identity_probe'
        or parameters.get('seed') != cfg.seed
        or tuple(parameters.get('c_grid', ())) != tuple(cfg.readout.c_grid)
        or parameters.get('cv_folds') != cfg.readout.cv_folds
        or parameters.get('tolerance') != 1e-6
        or output_path(path=Path(parameters.get('trace_source', '')), paths=paths) != source
    ):
        raise ValueError(f'{specification["id"]} differs from the frozen classifier protocol.')
    if manifest['status'] != 'completed':
        return {'status': manifest['status'], 'error': manifest.get('error'), 'manifest': manifest}
    report = json.loads(s=(directory / 'report.json').read_text(encoding='utf-8'))
    provenance = json.loads(s=(directory / 'provenance.json').read_text(encoding='utf-8'))
    if (
        report['parameters'] != parameters
        or provenance['membership_sha256'] != membership_sha256
        or provenance['source_trace'] != str(source.relative_to(paths.home))
        or report['model']['iterations'][0] > MAX_ITERATIONS
    ):
        raise ValueError(f'{specification["id"]} report and trace provenance disagree.')
    data = load_noise_trace(
        cfg=cfg,
        paths=paths,
        source=source,
        precision='float32',
        history=specification['history'],
        features=specification['features'],
    )
    replay = replay_exported_predictions(
        data=data,
        directory=directory,
        expected_sha256=report['model']['weights_sha256'],
    )
    sample_rows = read_table(path=directory / 'samples.parquet')
    by_id = {sample.sample_id: sample for sample in data.samples}
    if (
        len(sample_rows) != len(by_id)
        or {row['sample_id'] for row in sample_rows} != set(by_id)
        or any(
            row['label'] != by_id[row['sample_id']].label
            or row['split'] != by_id[row['sample_id']].split
            or row['fit_label'] != (row['label'] if row['split'] == 'train' else None)
            for row in sample_rows
        )
    ):
        raise ValueError(f'{specification["id"]} trained or scored different sample membership.')
    return {
        'status': 'completed',
        'manifest': manifest,
        'report': report,
        'replay': replay,
        'validation': read_table(path=directory / 'validation-predictions.parquet'),
        'identities': {sample.sample_id: sample.identity for sample in data.samples},
        'statistics': json.loads(
            s=(directory / 'feature-statistics.json').read_text(encoding='utf-8')
        ),
    }


def analyze_readout_ablation(
    cfg: ExperimentConfig, paths: Paths, output: Path, schedule: dict[str, Any]
) -> dict[str, Any]:
    """Check all frozen fits and compare each with the original paired validation rows.

    :param cfg: Unchanged 20-identity source experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Data home containing all immutable trace and classifier attempts.
    :type paths: Paths
    :param output: Fresh immutable analysis attempt directory.
    :type output: Path
    :param schedule: Previously committed eight-case schedule and source hashes.
    :type schedule: dict[str, Any]
    :returns: Per-case fit/replay evidence, paired intervals and preregistered gate.
    :rtype: dict[str, Any]
    :raises ValueError: If sources, schedule, membership or model replay are inconsistent.
    """
    parameters = {
        'kind': 'readout_ablation_analysis',
        'schedule_sha256': sha256_obj(obj=schedule),
        'bootstrap_samples': BOOTSTRAP_RESAMPLES,
        'bootstrap_seed': cfg.seed,
        'bootstrap_namespace': BOOTSTRAP_NAMESPACE,
        'advancement_threshold_pp': MINIMUM_IMPROVEMENT_PP,
        'trainable_fly_parameters': [],
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        cases = schedule['cases']
        if (
            cfg.seed != 0
            or cfg.memory.mode != 'persistent'
            or cfg.readout.cv_folds != 5
            or tuple(cfg.readout.c_grid) != (0.01, 0.1, 1.0, 10.0)
            or [
                (case['id'], case['features'], case['history'], case['components'])
                for case in cases
            ]
            != list(EXPECTED_CASES)
            or schedule['train_per_class'] != TRAIN_PER_CLASS
            or schedule['subset_seed'] != 0
            or schedule['trace_precision'] != 'float32'
            or schedule['bootstrap_resamples'] != BOOTSTRAP_RESAMPLES
            or schedule['bootstrap_seed'] != cfg.seed
            or schedule['advancement_minimum_percentage_points'] != MINIMUM_IMPROVEMENT_PP
        ):
            raise ValueError('Configuration or schedule differs from the frozen readout screen.')
        source = output_path(path=Path(schedule['source_trace']), paths=paths)
        reference = output_path(path=Path(schedule['reference_probe']), paths=paths)
        robustness = output_path(path=Path(schedule['robustness_trace']), paths=paths)
        for label, source_path, digest in (
            ('source', source, schedule['source_trace_inventory_sha256']),
            ('reference', reference, schedule['reference_probe_inventory_sha256']),
            ('robustness', robustness, schedule['robustness_trace_inventory_sha256']),
        ):
            if (
                sha256_file(path=source_path / 'checksums.sha256') != digest
                or verify_attempt_inventory(directory=source_path, paths=paths)['status']
                != 'completed'
            ):
                raise ValueError(f'Frozen {label} inventory differs from preregistration.')
        source_provenance = json.loads(s=(source / 'provenance.json').read_text(encoding='utf-8'))
        robustness_provenance = json.loads(
            s=(robustness / 'provenance.json').read_text(encoding='utf-8')
        )
        source_manifest = json.loads(s=(source / 'manifest.json').read_text(encoding='utf-8'))
        robustness_manifest = json.loads(
            s=(robustness / 'manifest.json').read_text(encoding='utf-8')
        )
        if (
            source_provenance['membership_sha256'] != schedule['membership_sha256']
            or robustness_provenance['membership_sha256'] != schedule['membership_sha256']
            or source_provenance['ordered_sample_ids_sha256']
            != robustness_provenance['ordered_sample_ids_sha256']
            or source_provenance['source_config_sha256']
            != robustness_provenance['source_config_sha256']
            or source_manifest['parameters']['episode_seed'] != 0
            or robustness_manifest['parameters']['episode_seed'] != 1
            or source_manifest['parameters']['episode_noise_enabled'] is not True
            or robustness_manifest['parameters']['episode_noise_enabled'] is not True
            or source_provenance['native_replay_exact'] is not True
        ):
            raise ValueError('Native trace sources differ from the frozen paired protocol.')
        evidence: dict[str, dict[str, Any]] = {}
        fits: dict[str, Any] = {}
        prefix = output_path(path=Path(schedule['working_prefix']), paths=paths)
        for specification in cases:
            case_id = specification['id']
            case_directory = reference if case_id == 'B0' else prefix / case_id
            record = _read_case(
                cfg=cfg,
                paths=paths,
                source=source,
                directory=case_directory,
                specification=specification,
                membership_sha256=schedule['membership_sha256'],
            )
            evidence[case_id] = record
            fits[case_id] = {
                'status': record['status'],
                'parameters': specification,
                'error': record.get('error'),
                'elapsed_seconds': record['manifest']['elapsed_seconds'],
                'rss_high_water_bytes': record['manifest']['rss_high_water_bytes'],
            }
            if record['status'] == 'completed':
                if case_id == 'B7' and (
                    record['manifest']['rss_high_water_bytes'] is None
                    or record['manifest']['rss_high_water_bytes'] > HISTORY_RSS_LIMIT_BYTES
                    or record['manifest']['elapsed_seconds'] > HISTORY_WALL_LIMIT_SECONDS
                ):
                    raise ValueError('All-history fit exceeded the preregistered resource budget.')
                fits[case_id].update(
                    cv_scores=record['report']['cv_scores'],
                    scores=record['report']['scores'],
                    model=record['report']['model'],
                    feature_statistics=record['statistics'],
                    numeric_replay=record['replay'],
                )
        baseline = evidence['B0']
        if baseline['status'] != 'completed':
            raise ValueError('The original reference readout must be completed.')
        baseline_rows = baseline['validation']
        ids = [row['sample_id'] for row in baseline_rows]
        truth = [row['y_true'] for row in baseline_rows]
        identities: NDArray = np.asarray(a=[baseline['identities'][sample_id] for sample_id in ids])
        reference_correct = np.asarray(
            a=[row['y_pred'] == row['y_true'] for row in baseline_rows], dtype=np.float64
        )
        paired: dict[str, Any] = {}
        for case_id in CASE_IDS[1:]:
            record = evidence[case_id]
            if record['status'] != 'completed':
                continue
            rows = record['validation']
            if [row['sample_id'] for row in rows] != ids or [
                row['y_true'] for row in rows
            ] != truth:
                raise ValueError(f'{case_id} validation images differ from B0.')
            candidate_correct = np.asarray(
                a=[row['y_pred'] == row['y_true'] for row in rows], dtype=np.float64
            )
            paired[case_id] = identity_cluster_interval(
                differences=candidate_correct - reference_correct,
                identities=identities,
                samples=BOOTSTRAP_RESAMPLES,
                seed=cfg.seed,
                namespace=BOOTSTRAP_NAMESPACE,
            )
        qualifying = [
            case_id
            for case_id in CASE_IDS[1:]
            if case_id in paired
            and paired[case_id]['diff_pp'] >= MINIMUM_IMPROVEMENT_PP
            and paired[case_id]['ci_low_pp'] > 0
        ]
        selected = (
            min(
                qualifying,
                key=lambda case_id: (
                    -paired[case_id]['diff_pp'],
                    fits[case_id]['scores']['validation']['log_loss'],
                    CASE_IDS.index(case_id),
                ),
            )
            if qualifying
            else None
        )
        report = {
            'parameters': parameters,
            'source_trace': schedule['source_trace'],
            'membership_sha256': schedule['membership_sha256'],
            'fits': fits,
            'paired_validation': paired,
            'selected_case': selected,
            'gate': 'advance_to_seed1' if selected is not None else 'do_not_advance',
            'inference': 'exploratory validation only; no historical test or reserve scoring',
            'memory_claim': False,
        }
        write_json(path=directory / 'report.json', value=report)
    return report

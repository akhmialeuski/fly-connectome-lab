"""Paired development-only recognition analysis for fixed episode-noise conditions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flystate.diagnostics.artifacts import attempt, verify_attempt_inventory
from flystate.evaluation.stats import bootstrap_mean_interval
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.hashing import sha256_obj
from flystate.settings import Paths, output_path
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table

NOISE_CASES: tuple[str, ...] = ('seed0', 'seed1', 'off')
TRACE_PRECISIONS: tuple[str, ...] = ('float32', 'float16')
BOOTSTRAP_SAMPLES: int = 2000
BOOTSTRAP_NAMESPACE: str = 'noise-recognition-identity-bootstrap'
PROMOTION_THRESHOLD_PP: float = 10.0
EXPECTED_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
EXPECTED_CV_FOLDS: int = 5
EXPECTED_EPISODE_STEPS: int = 16
EXPECTED_TRAIN_PER_CLASS: int = 14


def identity_cluster_interval(
    differences: NDArray[np.float64],
    identities: NDArray,
    samples: int,
    seed: int,
    namespace: str = BOOTSTRAP_NAMESPACE,
) -> dict[str, float | int]:
    """Bootstrap equally sized identity clusters of paired image-level differences.

    :param differences: Paired correctness differences (N,), possibly fractional.
    :type differences: NDArray[np.float64]
    :param identities: Identity labels (N,) in the same sample order.
    :type identities: NDArray
    :param samples: Positive resample count over identities.
    :type samples: int
    :param seed: Nonnegative config-derived resampling seed.
    :type seed: int
    :param namespace: Stable study-specific bootstrap seed namespace.
    :type namespace: str
    :returns: Difference and 95% percentile interval in percentage points.
    :rtype: dict[str, float | int]
    :raises ValueError: If vectors, finite values, or cluster sizes are invalid.
    """
    if (
        differences.ndim != 1
        or identities.shape != differences.shape
        or not len(differences)
        or not np.isfinite(differences).all()
    ):
        raise ValueError('Identity bootstrap requires matched finite observation vectors.')
    labels, inverse, counts = np.unique(ar=identities, return_inverse=True, return_counts=True)
    if not len(labels) or np.any(counts != counts[0]):
        raise ValueError('Identity bootstrap requires balanced nonempty validation clusters.')
    cluster_means = np.bincount(inverse, weights=differences) / counts
    low, high = bootstrap_mean_interval(
        values=cluster_means,
        samples=samples,
        seed=seed,
        namespace=namespace,
    )
    return {
        'diff_pp': float(np.mean(a=cluster_means) * 100),
        'ci_low_pp': low * 100,
        'ci_high_pp': high * 100,
        'clusters': len(labels),
    }


def _verified_probe(directory: Path, paths: Paths, trace: Path, precision: str) -> dict[str, Any]:
    """Verify a fixed neural readout attempt and return its evidence.

    :param directory: Completed or failed readout attempt.
    :type directory: Path
    :param paths: Working data boundary.
    :type paths: Paths
    :param trace: Expected native trace source.
    :type trace: Path
    :param precision: Expected float32 or float16 treatment.
    :type precision: str
    :returns: Manifest, report, validation rows, and sample identities when completed.
    :rtype: dict[str, Any]
    :raises ValueError: If a completed readout violates the frozen protocol.
    """
    source = output_path(path=directory, paths=paths)
    manifest = verify_attempt_inventory(directory=source, paths=paths)
    parameters = manifest['parameters']
    if (
        parameters.get('kind') != 'identity_probe'
        or parameters.get('representation') != 'neural'
        or parameters.get('history') != 'last'
        or parameters.get('features') != 'both'
        or parameters.get('label_mode') != 'true'
        or parameters.get('pca_components') != 60
        or parameters.get('max_iterations') != 50000
        or parameters.get('c_grid') != list(EXPECTED_C_GRID)
        or parameters.get('cv_folds') != EXPECTED_CV_FOLDS
        or parameters.get('tolerance') != 1e-6
        or parameters.get('seed') != 0
        or parameters.get('train_per_class') != EXPECTED_TRAIN_PER_CLASS
        or parameters.get('subset_seed') != 0
        or parameters.get('trace_precision') != precision
        or output_path(path=Path(parameters.get('trace_source', '')), paths=paths)
        != output_path(path=trace, paths=paths)
    ):
        raise ValueError('Readout attempt differs from the frozen neural protocol.')
    if manifest['status'] != 'completed':
        return {'status': manifest['status'], 'error': manifest.get('error')}
    report = json.loads(s=(source / 'report.json').read_text(encoding='utf-8'))
    provenance = json.loads(s=(source / 'provenance.json').read_text(encoding='utf-8'))
    if (
        report['parameters'] != parameters
        or report['model']['iterations'][0] > parameters['max_iterations']
        or provenance['trace_precision'] != precision
        or provenance['source_trace']
        != str(output_path(path=trace, paths=paths).relative_to(paths.home))
    ):
        raise ValueError('Completed readout report or provenance is inconsistent.')
    validation = read_table(path=source / 'validation-predictions.parquet')
    samples = read_table(path=source / 'samples.parquet')
    by_id = {row['sample_id']: row for row in samples if row['split'] == 'val'}
    if (
        len(by_id) != len(validation)
        or {row['sample_id'] for row in validation} != set(by_id)
        or any(row['y_true'] != by_id[row['sample_id']]['label'] for row in validation)
    ):
        raise ValueError('Validation prediction and sample membership disagree.')
    return {
        'status': 'completed',
        'report': report,
        'validation': validation,
        'identities': [by_id[row['sample_id']]['identity'] for row in validation],
        'directory': source,
    }


def _check_historical_replay(
    baseline: dict[str, Any], historical: Path, paths: Paths
) -> dict[str, Any]:
    """Require exact exported model arrays and predictions from the prior full-data endpoint.

    :param baseline: New seed-zero float16 readout evidence.
    :type baseline: dict[str, Any]
    :param historical: Immutable prior full-data neural diagnostic attempt.
    :type historical: Path
    :param paths: Working data boundary.
    :type paths: Paths
    :returns: Historical source identity and exact replay confirmation.
    :rtype: dict[str, Any]
    :raises ValueError: If the new baseline differs from the frozen prior endpoint.
    """
    old = output_path(path=historical, paths=paths)
    manifest = verify_attempt_inventory(directory=old, paths=paths)
    if manifest['status'] != 'completed':
        raise ValueError('Historical full-data neural endpoint is not completed.')
    current = baseline['directory']
    with (
        np.load(file=current / 'model' / 'weights.npz', allow_pickle=False) as fresh,
        np.load(file=old / 'model' / 'weights.npz', allow_pickle=False) as previous,
    ):
        if set(fresh.files) != set(previous.files) or any(
            not np.array_equal(a1=fresh[name], a2=previous[name]) for name in fresh.files
        ):
            raise ValueError('Seed-zero float16 classifier arrays do not replay historical arrays.')
    for split in ('train', 'validation'):
        if read_table(path=current / f'{split}-predictions.parquet') != read_table(
            path=old / f'{split}-predictions.parquet'
        ):
            raise ValueError(f'Seed-zero float16 {split} predictions differ from history.')
    return {'historical_attempt': str(old.relative_to(paths.home)), 'exact_replay': True}


def analyze_noise_recognition(
    cfg: ExperimentConfig, paths: Paths, output: Path, schedule: dict[str, Any]
) -> dict[str, Any]:
    """Compare six frozen readouts with paired validation identity-cluster intervals.

    :param cfg: Original cohort and seed-zero classifier configuration.
    :type cfg: ExperimentConfig
    :param paths: Existing trace and readout attempt home.
    :type paths: Paths
    :param output: Fresh immutable analysis attempt directory.
    :type output: Path
    :param schedule: Frozen paths, membership hash, and historical endpoint.
    :type schedule: dict[str, Any]
    :returns: Exact replay, six fits, paired intervals, and a screening-gate decision.
    :rtype: dict[str, Any]
    :raises ValueError: If source evidence, pairing, or baseline replay is invalid.
    """
    parameters = {
        'kind': 'noise_recognition_analysis',
        'schedule_sha256': sha256_obj(obj=schedule),
        'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'bootstrap_seed': cfg.seed,
        'bootstrap_namespace': BOOTSTRAP_NAMESPACE,
        'promotion_threshold_pp': PROMOTION_THRESHOLD_PP,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if (
            cfg.seed != 0
            or tuple(cfg.readout.c_grid) != EXPECTED_C_GRID
            or cfg.readout.cv_folds != EXPECTED_CV_FOLDS
            or cfg.episodes.steps != EXPECTED_EPISODE_STEPS
            or cfg.memory.mode != 'persistent'
        ):
            raise ValueError('Configuration differs from the frozen noise-recognition protocol.')
        if set(schedule['traces']) != set(NOISE_CASES) or set(schedule['probes']) != set(
            NOISE_CASES
        ):
            raise ValueError('Schedule must contain every preregistered noise condition.')
        traces = {
            case: output_path(path=Path(schedule['traces'][case]), paths=paths)
            for case in NOISE_CASES
        }
        trace_sources: list[dict[str, Any]] = []
        for case, trace in traces.items():
            manifest = verify_attempt_inventory(directory=trace, paths=paths)
            provenance = json.loads(s=(trace / 'provenance.json').read_text(encoding='utf-8'))
            expected_seed = 1 if case == 'seed1' else 0
            if (
                manifest['status'] != 'completed'
                or manifest['parameters']['kind'] != 'noise_trace'
                or manifest['parameters']['episode_seed'] != expected_seed
                or manifest['parameters']['episode_noise_enabled'] != (case != 'off')
                or manifest['parameters']['rest_seed'] != cfg.seed
                or provenance['source_config_sha256'] != config_hash(cfg=cfg)
                or provenance['membership_sha256'] != schedule['membership_sha256']
                or (case == 'seed0' and provenance['native_replay_exact'] is not True)
            ):
                raise ValueError(f'Native {case} trace violates the frozen schedule.')
            trace_sources.append(provenance)
        if (
            len({p['dataset_fingerprint'] for p in trace_sources}) != 1
            or len({p['ordered_sample_ids_sha256'] for p in trace_sources}) != 1
        ):
            raise ValueError('Noise traces differ in images or ordered development samples.')
        probes: dict[str, dict[str, dict[str, Any]]] = {}
        summary: dict[str, Any] = {}
        for case in NOISE_CASES:
            if set(schedule['probes'][case]) != set(TRACE_PRECISIONS):
                raise ValueError('Every noise condition requires both storage precisions.')
            probes[case] = {}
            summary[case] = {}
            for precision in TRACE_PRECISIONS:
                evidence = _verified_probe(
                    directory=Path(schedule['probes'][case][precision]),
                    paths=paths,
                    trace=traces[case],
                    precision=precision,
                )
                probes[case][precision] = evidence
                summary[case][precision] = (
                    {
                        'status': 'completed',
                        'cv_scores': evidence['report']['cv_scores'],
                        'scores': evidence['report']['scores'],
                        'model': evidence['report']['model'],
                    }
                    if evidence['status'] == 'completed'
                    else {'status': evidence['status'], 'error': evidence['error']}
                )
        baseline = probes['seed0']['float16']
        historical_replay = (
            _check_historical_replay(
                baseline=baseline,
                historical=Path(schedule['historical']),
                paths=paths,
            )
            if baseline['status'] == 'completed'
            else {'exact_replay': False, 'reason': 'baseline fit failed'}
        )
        paired: dict[str, Any] = {}
        precision_effects: dict[str, Any] = {}
        all_completed = all(
            probes[case][precision]['status'] == 'completed'
            for case in NOISE_CASES
            for precision in TRACE_PRECISIONS
        )
        if all_completed:
            for precision in TRACE_PRECISIONS:
                rows = {case: probes[case][precision]['validation'] for case in NOISE_CASES}
                ids = [row['sample_id'] for row in rows['seed0']]
                truth = [row['y_true'] for row in rows['seed0']]
                if any(
                    [row['sample_id'] for row in rows[case]] != ids
                    or [row['y_true'] for row in rows[case]] != truth
                    for case in NOISE_CASES
                ):
                    raise ValueError('Noise conditions must share ordered validation labels.')
                identities = np.asarray(a=probes['seed0'][precision]['identities'])
                correct = {
                    case: np.asarray(
                        a=[row['y_pred'] == row['y_true'] for row in rows[case]],
                        dtype=np.float64,
                    )
                    for case in NOISE_CASES
                }
                paired[precision] = {
                    case: identity_cluster_interval(
                        differences=correct['off'] - correct[case],
                        identities=identities,
                        samples=BOOTSTRAP_SAMPLES,
                        seed=cfg.seed,
                    )
                    for case in ('seed0', 'seed1')
                }
                paired[precision]['mean_noisy'] = identity_cluster_interval(
                    differences=correct['off'] - (correct['seed0'] + correct['seed1']) / 2,
                    identities=identities,
                    samples=BOOTSTRAP_SAMPLES,
                    seed=cfg.seed,
                )
            for case in NOISE_CASES:
                first = probes[case]['float32']['validation']
                second = probes[case]['float16']['validation']
                if [row['sample_id'] for row in first] != [row['sample_id'] for row in second]:
                    raise ValueError('Storage-precision conditions must share ordered images.')
                precision_effects[case] = {
                    'prediction_disagreements': sum(
                        a['y_pred'] != b['y_pred'] for a, b in zip(first, second, strict=True)
                    ),
                    'maximum_probability_difference': max(
                        abs(x - y)
                        for a, b in zip(first, second, strict=True)
                        for x, y in zip(a['probabilities'], b['probabilities'], strict=True)
                    ),
                }
        gate = (
            'promote_noise_off'
            if all_completed
            and all(
                paired[precision]['mean_noisy']['diff_pp'] >= PROMOTION_THRESHOLD_PP
                and paired[precision]['mean_noisy']['ci_low_pp'] > 0
                for precision in TRACE_PRECISIONS
            )
            else 'do_not_promote'
            if all_completed
            else 'inconclusive_failed_fit'
        )
        report = {
            'parameters': parameters,
            'membership_sha256': schedule['membership_sha256'],
            'historical_replay': historical_replay,
            'fits': summary,
            'paired_validation': paired,
            'precision_effects': precision_effects,
            'gate': gate,
            'inference': 'exploratory validation only; no historical test or reserve scoring',
            'memory_claim': False,
        }
        write_json(path=directory / 'report.json', value=report)
    return report

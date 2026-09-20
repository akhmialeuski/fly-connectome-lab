"""Strict paired comparisons of completed evaluations with explicit compatibility checks."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from flystate.evaluation.stats import mcnemar, paired_bootstrap_diff
from flystate.experiments.config import ConfigError, config_hash, load_config
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table
from flystate.storage.runs import RunError, load_manifest


class IncompatibleRunsError(ValueError):
    """Evaluated runs cannot support the requested paired scientific comparison."""


def flatten_config(values: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    """Flatten nested configuration mappings while retaining lists as whole values.

    :param values: Effective configuration mapping.
    :type values: dict[str, Any]
    :param prefix: Internal dotted prefix for recursive traversal.
    :type prefix: str
    :returns: Exact dotted leaf values.
    :rtype: dict[str, Any]
    """
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(values=value, prefix=name))
        else:
            flattened[name] = value
    return flattened


def load_evaluation(
    run_dir: Path, evaluation_id: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a selected evaluation or the latest completed test evaluation.

    :param run_dir: Completed training directory.
    :type run_dir: Path
    :param evaluation_id: Optional basename of a specific evaluation.
    :type evaluation_id: Optional[str]
    :returns: Evaluation metadata and prediction records.
    :rtype: tuple[dict[str, Any], list[dict[str, Any]]]
    :raises RunError: If selection, lifecycle, or evaluation identity is invalid.
    """
    parent = run_dir / 'evals'
    if evaluation_id is not None:
        if Path(evaluation_id).name != evaluation_id or evaluation_id in {'.', '..'}:
            raise RunError('Evaluation identifier must be a directory basename.')
        candidates = [parent / evaluation_id]
    else:
        candidates = sorted(parent.glob('*'), reverse=True)
    for directory in candidates:
        metadata = json.loads(s=(directory / 'eval.json').read_text(encoding='utf-8'))
        if evaluation_id is None and (
            metadata.get('status') != 'completed' or metadata.get('split') != 'test'
        ):
            continue
        if (
            metadata.get('status') != 'completed'
            or metadata.get('run_id') != run_dir.name
            or metadata.get('eval_id') != directory.name
        ):
            raise RunError('Evaluation is incomplete or its recorded identity is inconsistent.')
        return metadata, read_table(path=directory / 'predictions.parquet')
    raise RunError('Run has no completed test evaluation; run flystate evaluate RUN.')


def _prediction_map(
    metadata: dict[str, Any], predictions: list[dict[str, Any]], name: str
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate unique complete sample-observation pairs and consistent class identities.

    :param metadata: Evaluation dimensions.
    :type metadata: dict[str, Any]
    :param predictions: Stored prediction records.
    :type predictions: list[dict[str, Any]]
    :param name: Diagnostic run label.
    :type name: str
    :returns: Validated records keyed by sample identity and observation.
    :rtype: dict[tuple[str, int], dict[str, Any]]
    :raises IncompatibleRunsError: If the prediction grid is incomplete or internally inconsistent.
    """
    try:
        steps = metadata['timesteps']
        n = metadata['n_samples']
        if type(steps) is not int or type(n) is not int or min(steps, n) < 1:
            raise ValueError('invalid evaluation dimensions')
        mapping: dict[tuple[str, int], dict[str, Any]] = {}
        labels: dict[str, int] = {}
        for row in predictions:
            sample = row['sample_id']
            step = row['t']
            if not isinstance(sample, str) or type(step) is not int or not 1 <= step <= steps:
                raise ValueError('invalid sample or observation identity')
            key = (sample, step)
            if key in mapping:
                raise ValueError('duplicate sample-observation prediction')
            if (
                type(row['y_true']) is not int
                or type(row['y_pred']) is not int
                or type(row['correct']) is not bool
            ):
                raise ValueError('invalid label or correctness types')
            if row['correct'] != (row['y_true'] == row['y_pred']):
                raise ValueError('correctness does not match stored labels')
            if sample in labels and labels[sample] != row['y_true']:
                raise ValueError('sample labels change between observations')
            labels[sample] = row['y_true']
            mapping[key] = row
        if len(labels) != n or len(mapping) != n * steps:
            raise ValueError('incomplete sample-observation prediction grid')
        return mapping
    except (KeyError, TypeError, ValueError) as error:
        raise IncompatibleRunsError(f'{name}: {error}') from error


def compare(
    run_a: Path,
    run_b: Path,
    paths: Paths,
    eval_a: str | None = None,
    eval_b: str | None = None,
    allowed: tuple[str, ...] = ('memory.mode', 'name'),
    bootstrap: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compare paired per-observation correctness after checking all scientific identities.

    :param run_a: First completed training directory.
    :type run_a: Path
    :param run_b: Second completed training directory.
    :type run_b: Path
    :param paths: Experiment storage home.
    :type paths: Paths
    :param eval_a: Optional explicit evaluation for A.
    :type eval_a: Optional[str]
    :param eval_b: Optional explicit evaluation for B.
    :type eval_b: Optional[str]
    :param allowed: Exact dotted configuration differences explicitly permitted.
    :type allowed: tuple[str, ...]
    :param bootstrap: Optional positive resample count, defaulting to run A.
    :type bootstrap: Optional[int]
    :param seed: Optional nonnegative bootstrap seed, defaulting to run A.
    :type seed: Optional[int]
    :returns: Published comparison metadata and paired statistics.
    :rtype: dict[str, Any]
    :raises IncompatibleRunsError: If any immutable identity or unapproved configuration differs.
    :raises RunError: If a run is incomplete or its configuration hash is invalid.
    :raises ConfigError: If allowed keys or bootstrap settings are invalid.
    """
    manifests = [load_manifest(run_dir=directory) for directory in (run_a, run_b)]
    configs = [load_config(path=directory / 'config.yaml') for directory in (run_a, run_b)]
    for manifest, cfg in zip(manifests, configs, strict=True):
        if manifest['status'] != 'completed' or manifest.get('config_hash') != config_hash(cfg=cfg):
            raise RunError(
                'Comparison requires completed runs with intact effective configurations.'
            )
    flat_a, flat_b = [flatten_config(values=cfg.model_dump(mode='json')) for cfg in configs]
    unknown = sorted(set(allowed) - flat_a.keys())
    if unknown:
        raise ConfigError(f'Unknown allowed configuration keys: {unknown}')
    samples = configs[0].evaluation.bootstrap_samples if bootstrap is None else bootstrap
    bootstrap_seed = configs[0].evaluation.bootstrap_seed if seed is None else seed
    if samples < 1 or bootstrap_seed < 0:
        raise ConfigError('Bootstrap count must be positive and seed nonnegative.')
    meta_a, predictions_a = load_evaluation(run_dir=run_a, evaluation_id=eval_a)
    meta_b, predictions_b = load_evaluation(run_dir=run_b, evaluation_id=eval_b)
    differences = {
        key: {'a': flat_a[key], 'b': flat_b[key]} for key in flat_a if flat_a[key] != flat_b[key]
    }
    errors = [f'config.{key}: {value}' for key, value in differences.items() if key not in allowed]
    for field in ('dataset_fingerprint', 'trajectory_hash', 'brain_files_sha256', 'threads'):
        if manifests[0].get(field) is None or manifests[0].get(field) != manifests[1].get(field):
            errors.append(
                f'manifest.{field}: {manifests[0].get(field)!r} != {manifests[1].get(field)!r}'
            )
    for field in ('split', 'timesteps', 'n_classes'):
        if meta_a.get(field) != meta_b.get(field):
            errors.append(f'evaluation.{field}: {meta_a.get(field)!r} != {meta_b.get(field)!r}')
    maps = []
    for name, metadata, records in (('A', meta_a, predictions_a), ('B', meta_b, predictions_b)):
        try:
            maps.append(_prediction_map(metadata=metadata, predictions=records, name=name))
        except IncompatibleRunsError as error:
            errors.append(str(error))
            maps.append({})
    pairs_a, pairs_b = maps
    if pairs_a.keys() != pairs_b.keys():
        errors.append('prediction sample-observation identities differ')
    if any(
        pairs_a[key]['y_true'] != pairs_b[key]['y_true'] for key in pairs_a.keys() & pairs_b.keys()
    ):
        errors.append('prediction true labels differ for paired samples')
    if errors:
        raise IncompatibleRunsError('\n'.join(errors))
    sample_ids = sorted({key[0] for key in pairs_a})
    rows = []
    for step in range(1, meta_a['timesteps'] + 1):
        correct_a = np.asarray(
            a=[pairs_a[(sample, step)]['correct'] for sample in sample_ids], dtype=np.bool_
        )
        correct_b = np.asarray(
            a=[pairs_b[(sample, step)]['correct'] for sample in sample_ids], dtype=np.bool_
        )
        rows.append(
            {
                't': step,
                'acc_a': float(correct_a.mean()),
                'acc_b': float(correct_b.mean()),
                **paired_bootstrap_diff(
                    correct_a=correct_a, correct_b=correct_b, samples=samples, seed=bootstrap_seed
                ),
                **mcnemar(
                    n01=int(np.sum(correct_a & ~correct_b)), n10=int(np.sum(~correct_a & correct_b))
                ),
            }
        )
    delta = None
    if configs[0].memory.mode == 'persistent' and configs[1].memory.mode == 'reset':
        delta = {key: rows[-1][key] for key in ('diff_pp', 'ci_low_pp', 'ci_high_pp', 'p')}
    now = datetime.now(tz=UTC)
    result = {
        'schema_version': 1,
        'created_utc': now.isoformat(),
        'run_a': run_a.name,
        'run_b': run_b.name,
        'eval_a': meta_a['eval_id'],
        'eval_b': meta_b['eval_id'],
        'split': meta_a['split'],
        'allowed_differences': differences,
        'bootstrap': samples,
        'seed': bootstrap_seed,
        'rows': rows,
        'delta_mem_pp': delta,
    }
    destination = (
        paths.runs
        / 'comparisons'
        / f'{now:%Y%m%d-%H%M%S}-{run_a.name}-vs-{run_b.name}-{uuid4().hex[:8]}.json'
    )
    write_json(path=destination, value=result)
    return result


def comparison_markdown(result: dict[str, Any]) -> str:
    """Render a comparison report with explicit run identities and paired intervals.

    :param result: Validated comparison result.
    :type result: dict[str, Any]
    :returns: English Markdown report ending in a newline.
    :rtype: str
    """
    lines = [
        f'# {result["run_a"]} vs {result["run_b"]}',
        '',
        f'Split: {result["split"]}. Bootstrap resamples: {result["bootstrap"]}; '
        f'seed: {result["seed"]}.',
        '',
        '| t | Accuracy A | Accuracy B | Difference (pp) | 95% CI (pp) | '
        'A only | B only | p | Method |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |',
    ]
    for row in result['rows']:
        lines.append(
            f'| {row["t"]} | {row["acc_a"]:.2%} | {row["acc_b"]:.2%} | {row["diff_pp"]:.2f} | '
            f'[{row["ci_low_pp"]:.2f}, {row["ci_high_pp"]:.2f}] | {row["n01"]} | {row["n10"]} | '
            f'{row["p"]:.6g} | {row["method"]} |'
        )
    if result['delta_mem_pp'] is not None:
        delta = result['delta_mem_pp']
        lines.extend(
            [
                '',
                f'Final memory effect (persistent minus reset): {delta["diff_pp"]:.2f} pp; '
                f'95% paired interval [{delta["ci_low_pp"]:.2f}, {delta["ci_high_pp"]:.2f}] pp; '
                f'McNemar p={delta["p"]:.6g}.',
            ]
        )
    return '\n'.join(lines) + '\n'

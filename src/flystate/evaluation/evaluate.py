"""Append-only evaluation of immutable runs using their recorded trace identity."""

import json
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow as pa
from threadpoolctl import threadpool_limits

from flystate import __version__
from flystate.episodes.episode import EpisodeBuilder
from flystate.evaluation.stats import wilson_interval
from flystate.experiments.config import ExperimentConfig, config_hash, load_config
from flystate.readouts.linear import LinearReadout, features_at, load_readouts
from flystate.readouts.training import accuracy_metrics, read_feature_rows
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table
from flystate.storage.runs import RunError, load_manifest
from flystate.traces.key import simulation_mode
from flystate.traces.store import TraceStore

PREDICTION_SCHEMA: pa.Schema = pa.schema(
    fields=[
        ('sample_id', pa.string()),
        ('t', pa.int16()),
        ('y_true', pa.int32()),
        ('y_pred', pa.int32()),
        ('p_pred', pa.float32()),
        ('p_true', pa.float32()),
        ('top5_labels', pa.list_(value_type=pa.int32())),
        ('top5_probs', pa.list_(value_type=pa.float32())),
        ('correct', pa.bool_()),
    ]
)


def evaluation_inputs(
    run_dir: Path, paths: Paths
) -> tuple[ExperimentConfig, TraceStore, list[LinearReadout]]:
    """Validate recorded artifacts without resolving datasets or automatic threads again.

    :param run_dir: Completed training directory.
    :type run_dir: Path
    :param paths: Storage installation containing the recorded cache.
    :type paths: Paths
    :returns: Effective configuration, verified cache, and ordered portable models.
    :rtype: tuple[ExperimentConfig, TraceStore, list[LinearReadout]]
    :raises RunError: If run status, provenance, or model geometry is inconsistent.
    """
    manifest = load_manifest(run_dir=run_dir)
    if manifest['status'] != 'completed':
        raise RunError('Evaluation requires a completed training run.')
    cfg = load_config(path=run_dir / 'config.yaml')
    if config_hash(cfg=cfg) != manifest.get('config_hash'):
        raise RunError('Effective configuration does not match the recorded hash.')
    key = manifest.get('cache_key')
    if not isinstance(key, str) or re.fullmatch(pattern=r'[0-9a-f]{64}', string=key) is None:
        raise RunError('Recorded cache key must be a SHA-256 digest.')
    store = TraceStore(directory=paths.features / key)
    store.verify(require_complete=True)
    fields = store.meta['key_fields']
    brain = cfg.brain.model_dump(mode='json', exclude={'threads', 'batch_size'})
    brain['threads'] = manifest.get('threads')
    expected = {
        'dataset': cfg.dataset.model_dump(mode='json'),
        'episodes': cfg.episodes.model_dump(mode='json'),
        'encoder': cfg.encoder.model_dump(mode='json'),
        'brain': brain,
        'simulation_mode': simulation_mode(cfg=cfg),
        'readout': cfg.readout.model_dump(
            mode='json', include={'population', 'features', 'trace_tau_s'}
        ),
        'seed': cfg.seed,
        'dataset_fingerprint': manifest.get('dataset_fingerprint'),
        'brain_files_sha256': manifest.get('brain_files_sha256'),
        'flybrain_version': manifest.get('flybrain_version'),
    }
    if any(fields.get(name) != value for name, value in expected.items()):
        raise RunError('Recorded trace identity does not match the training run.')
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    if builder.trajectory_hash(
        sample_ids=[row['sample_id'] for row in store.index]
    ) != manifest.get('trajectory_hash'):
        raise RunError('Recorded trajectory hash does not match the trace sample identities.')
    models = load_readouts(path=run_dir / 'model')
    metadata = json.loads(s=(run_dir / 'model' / 'model.json').read_text(encoding='utf-8'))
    classes = np.unique(ar=[row['label'] for row in store.index if row['split'] == 'train'])
    if (
        len(models) != store.steps
        or store.steps != cfg.episodes.steps
        or metadata.get('mode') != cfg.memory.mode
        or metadata.get('cache_key') != key
        or any(
            not np.array_equal(a1=model.classes, a2=classes)
            or len(model.scaler_mean)
            != store.features_count * (step if cfg.memory.mode == 'reset_concat' else 1)
            for step, model in enumerate(models, start=1)
        )
    ):
        raise RunError('Readout geometry, classes, mode, or cache identity is inconsistent.')
    return cfg, store, models


def _reserve_evaluation(run_dir: Path, split: str) -> tuple[Path, str]:
    """Allocate a collision-safe directory without changing existing run artifacts.

    :param run_dir: Completed training directory.
    :type run_dir: Path
    :param split: Validated evaluation split.
    :type split: str
    :returns: Reserved directory and ISO creation timestamp.
    :rtype: tuple[Path, str]
    """
    now = datetime.now(tz=UTC)
    stem = f'{now:%Y%m%d-%H%M%S}-{split}'
    parent = run_dir / 'evals'
    parent.mkdir(exist_ok=True)
    suffix = 1
    while True:
        directory = parent / (stem if suffix == 1 else f'{stem}-{suffix:06}')
        try:
            directory.mkdir()
            return directory, now.isoformat()
        except FileExistsError:
            suffix += 1


def evaluate(run_dir: Path, paths: Paths, split: str = 'test') -> dict[str, Any]:
    """Evaluate every observation and publish typed predictions in a fresh subdirectory.

    :param run_dir: Completed training run.
    :type run_dir: Path
    :param paths: Experiment storage home.
    :type paths: Paths
    :param split: Test for final reporting or val for validation reproduction.
    :type split: str
    :returns: Evaluation identity, per-observation metrics, and accuracy mapping.
    :rtype: dict[str, Any]
    :raises ValueError: If the split is unsupported or contains no samples.
    """
    if split not in {'test', 'val'}:
        raise ValueError('Evaluation split must be test or val.')
    cfg, store, models = evaluation_inputs(run_dir=run_dir, paths=paths)
    rows = np.asarray(
        a=[row['row'] for row in store.index if row['split'] == split], dtype=np.int64
    )
    if not len(rows):
        raise ValueError(f'No samples exist for evaluation split {split}.')
    samples = [store.index[int(row)] for row in rows]
    targets = np.asarray(a=[row['label'] for row in samples], dtype=np.int64)
    directory, created = _reserve_evaluation(run_dir=run_dir, split=split)
    metadata = {
        'schema_version': 1,
        'eval_id': directory.name,
        'run_id': run_dir.name,
        'split': split,
        'created_utc': created,
        'n_samples': len(rows),
        'n_classes': len(models[0].classes),
        'timesteps': len(models),
        'versions': {
            'flystate': __version__,
            'scikit-learn': version(distribution_name='scikit-learn'),
        },
        'status': 'running',
    }
    write_json(path=directory / 'eval.json', value=metadata)
    try:
        features = read_feature_rows(store=store, rows=rows)
        predictions: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        elapsed = 0.0
        with threadpool_limits(limits=1, user_api='blas'):
            for step, model in enumerate(models, start=1):
                started = perf_counter()
                probabilities = model.predict_proba(
                    x=features_at(features=features, t=step, mode=cfg.memory.mode)
                )
                elapsed += perf_counter() - started
                ordering = np.argsort(a=-probabilities, axis=1, kind='stable')
                predicted = model.classes[ordering[:, 0]]
                low, high = wilson_interval(
                    successes=int(np.sum(predicted == targets)), n=len(rows)
                )
                metrics.append(
                    {
                        't': step,
                        **accuracy_metrics(
                            probabilities=probabilities, targets=targets, classes=model.classes
                        ),
                        'ci_low': low,
                        'ci_high': high,
                    }
                )
                true_columns = np.searchsorted(a=model.classes, v=targets)
                for index, sample in enumerate(samples):
                    top = ordering[index, : min(5, len(model.classes))]
                    predictions.append(
                        {
                            'sample_id': sample['sample_id'],
                            't': step,
                            'y_true': int(targets[index]),
                            'y_pred': int(predicted[index]),
                            'p_pred': float(probabilities[index, ordering[index, 0]]),
                            'p_true': float(probabilities[index, true_columns[index]]),
                            'top5_labels': model.classes[top].tolist(),
                            'top5_probs': probabilities[index, top].tolist(),
                            'correct': bool(predicted[index] == targets[index]),
                        }
                    )
        pairs, counts = np.unique(
            ar=np.asarray(a=[(row['y_true'], row['y_pred']) for row in predictions[-len(rows) :]]),
            axis=0,
            return_counts=True,
        )
        confusion = [
            {'y_true': int(pair[0]), 'y_pred': int(pair[1]), 'count': int(count)}
            for pair, count in zip(pairs, counts, strict=True)
        ]
        brain_ms = np.asarray(a=store.array(name='brain_ms').oindex[rows])
        timings = [
            {
                'sample_id': sample['sample_id'],
                'brain_ms': float(brain_ms[index]),
                'readout_ms': elapsed * 1000 / len(rows),
            }
            for index, sample in enumerate(samples)
        ]
        write_table(
            path=directory / 'predictions.parquet', rows=predictions, schema=PREDICTION_SCHEMA
        )
        write_table(path=directory / 'metrics.parquet', rows=metrics)
        write_table(path=directory / 'confusion.parquet', rows=confusion)
        write_table(path=directory / 'timings.parquet', rows=timings)
        write_json(path=directory / 'eval.json', value={**metadata, 'status': 'completed'})
    except BaseException as error:
        write_json(
            path=directory / 'eval.json',
            value={**metadata, 'status': 'failed', 'error': str(error)},
        )
        raise
    return {
        'eval_id': directory.name,
        'eval_dir': str(directory),
        'accuracy_by_t': {str(row['t']): row['accuracy'] for row in metrics},
        'rows': metrics,
    }

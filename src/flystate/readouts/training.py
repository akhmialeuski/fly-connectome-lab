"""Per-observation training from completed caches with validation-only model selection."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ExperimentConfig
from flystate.log import configure_logging, get_logger
from flystate.readouts.linear import LinearReadout, features_at, fit_readout, save_readouts
from flystate.settings import Paths
from flystate.storage.runs import (
    create_run,
    load_manifest,
    update_manifest,
    write_run_json,
    write_run_parquet,
)
from flystate.traces.builder import open_trace
from flystate.traces.store import TraceError, TraceStore


def read_feature_rows(store: TraceStore, rows: NDArray) -> NDArray[np.float32]:
    """Read only requested logical rows in bounded Zarr selections.

    :param store: Verified completed trace cache.
    :type store: TraceStore
    :param rows: Unique increasing integer row indices, shape (N,).
    :type rows: NDArray
    :returns: Float32 selected episode features (N,T,F).
    :rtype: NDArray[np.float32]
    :raises ValueError: If row indices are malformed, repeated, or outside the cache.
    """
    if (
        rows.ndim != 1
        or rows.dtype.kind not in 'iu'
        or np.any(rows < 0)
        or np.any(rows >= store.n)
        or np.any(rows[1:] <= rows[:-1])
    ):
        raise ValueError('Feature rows must be increasing unique integer indices within the cache.')
    values = np.empty(shape=(len(rows), store.steps, store.features_count), dtype=np.float32)
    for start in range(0, len(rows), store.chunk_size):
        selected = rows[start : start + store.chunk_size]
        values[start : start + len(selected)] = np.asarray(
            a=store.array(name='features').oindex[selected, :, :], dtype=np.float32
        )
    return values


def accuracy_metrics(
    probabilities: NDArray, targets: NDArray, classes: NDArray
) -> dict[str, float | int]:
    """Compute top-one and top-five accuracies with deterministic tie ordering.

    :param probabilities: Finite class probabilities (N,C).
    :type probabilities: NDArray
    :param targets: Integer class labels (N,).
    :type targets: NDArray
    :param classes: Ordered class labels corresponding to probability columns.
    :type classes: NDArray
    :returns: Accuracy, top-five accuracy using min(5,C), and image count.
    :rtype: dict[str, float | int]
    :raises ValueError: If prediction shapes, probabilities, or targets are invalid.
    """
    if (
        probabilities.ndim != 2
        or not len(probabilities)
        or targets.ndim != 1
        or classes.ndim != 1
        or len(classes) < 2
        or len(np.unique(ar=classes)) != len(classes)
        or probabilities.shape != (len(targets), len(classes))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(a=probabilities.sum(axis=1), b=1, rtol=1e-6, atol=1e-8)
        or not np.isin(element=targets, test_elements=classes).all()
    ):
        raise ValueError('Predictions and targets must share nonempty rows and known classes.')
    ordering = np.argsort(a=-probabilities, axis=1, kind='stable')
    predictions = classes[ordering[:, 0]]
    top = classes[ordering[:, : min(5, len(classes))]]
    return {
        'accuracy': float(np.mean(a=predictions == targets)),
        'top5_accuracy': float(np.mean(a=np.any(top == targets[:, None], axis=1))),
        'n': len(targets),
    }


def train(
    cfg: ExperimentConfig, paths: Paths, original_yaml: str, verbosity: int = 0
) -> dict[str, Any]:
    """Fit one readout per observation and atomically complete a unique immutable run.

    :param cfg: Effective validated experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage home.
    :type paths: Paths
    :param original_yaml: Exact input configuration text before overrides.
    :type original_yaml: str
    :param verbosity: Console log verbosity to preserve across run logging.
    :type verbosity: int
    :returns: Run identity, location, and validation summary.
    :rtype: dict[str, Any]
    """
    try:
        store = open_trace(cfg=cfg, paths=paths, require_complete=True)
    except TraceError as error:
        raise TraceError(
            f'Completed cache required; run flystate trace build CONFIG. {error}'
        ) from error
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
    run_dir = create_run(
        paths=paths,
        cfg=cfg,
        original_yaml=original_yaml,
        extra_manifest={
            'cache_key': store.meta['key'],
            'dataset_fingerprint': store.meta['key_fields']['dataset_fingerprint'],
            'trajectory_hash': builder.trajectory_hash(
                sample_ids=[row['sample_id'] for row in store.index]
            ),
            'threads': store.meta['threads'],
            'brain_files_sha256': store.meta['key_fields']['brain_files_sha256'],
        },
    )
    try:
        configure_logging(verbosity=verbosity, json_path=run_dir / 'logs' / 'events.jsonl')
        train_rows = np.asarray(
            a=[row['row'] for row in store.index if row['split'] == 'train'], dtype=np.int64
        )
        val_rows = np.asarray(
            a=[row['row'] for row in store.index if row['split'] == 'val'], dtype=np.int64
        )
        labels = np.asarray(a=[row['label'] for row in store.index], dtype=np.int64)
        training = read_feature_rows(store=store, rows=train_rows)
        validation = read_feature_rows(store=store, rows=val_rows)
        readouts: list[LinearReadout] = []
        metrics: list[dict[str, Any]] = []
        cv_scores = {}
        with threadpool_limits(limits=1, user_api='blas'):
            for step in range(1, store.steps + 1):
                get_logger(name='train').info(
                    'fit_readout', run_id=run_dir.name, phase='training', t=step
                )
                model, scores = fit_readout(
                    x_train=features_at(features=training, t=step, mode=cfg.memory.mode),
                    y_train=labels[train_rows],
                    cfg=cfg.readout,
                    seed=cfg.seed,
                )
                probabilities = model.predict_proba(
                    x=features_at(features=validation, t=step, mode=cfg.memory.mode)
                )
                metrics.append(
                    {
                        't': step,
                        **accuracy_metrics(
                            probabilities=probabilities,
                            targets=labels[val_rows],
                            classes=model.classes,
                        ),
                        'c_value': model.c_value,
                        'pca_components': model.pca_components.shape[0],
                    }
                )
                cv_scores[str(step)] = scores
                readouts.append(model)
        save_readouts(
            path=run_dir / 'model',
            readouts=readouts,
            meta={'mode': cfg.memory.mode, 'cv_scores': cv_scores, 'cache_key': store.meta['key']},
        )
        write_run_parquet(run_dir=run_dir, relative='metrics/validation.parquet', rows=metrics)
        summary = {
            'final_val_accuracy': metrics[-1]['accuracy'],
            'val_accuracy_by_t': {str(row['t']): row['accuracy'] for row in metrics},
            'mode': cfg.memory.mode,
        }
        write_run_json(run_dir=run_dir, relative='summary.json', value=summary)
        get_logger(name='train').info(
            'training_completed',
            run_id=run_dir.name,
            phase='completed',
            final_val_accuracy=summary['final_val_accuracy'],
        )
        update_manifest(
            run_dir=run_dir, status='completed', finished_utc=datetime.now(tz=UTC).isoformat()
        )
    except BaseException as error:
        if load_manifest(run_dir=run_dir)['status'] != 'completed':
            update_manifest(
                run_dir=run_dir,
                status='failed',
                error=str(error),
                finished_utc=datetime.now(tz=UTC).isoformat(),
            )
        raise
    finally:
        configure_logging(verbosity=verbosity)
    return {'run_id': run_dir.name, 'run_dir': str(run_dir), **summary}

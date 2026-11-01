"""Pixel controls that measure whether an observation sequence adds information."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.datasets.preprocess import prepare_dataset
from flystate.episodes.episode import EpisodeBuilder
from flystate.evaluation.stats import bootstrap_mean_interval, wilson_interval
from flystate.experiments.config import ExperimentConfig, config_hash
from flystate.log import get_logger
from flystate.readouts.fitting import fit_classifier
from flystate.settings import Paths
from flystate.storage.json import write_json

PIXEL_PCA_COMPONENTS: int = 100
PIXEL_MAX: float = 255.0


def _baseline(
    features: NDArray, labels: NDArray, train: NDArray, evaluation: NDArray, cfg: ExperimentConfig
) -> tuple[dict[str, Any], NDArray]:
    """Fit a pixel classifier and score independent evaluation images.

    :param features: Float32 normalized pixels, shape (N,F).
    :type features: NDArray
    :param labels: Integer targets, shape (N,).
    :type labels: NDArray
    :param train: Training row indices.
    :type train: NDArray
    :param evaluation: Evaluation row indices.
    :type evaluation: NDArray
    :param cfg: Validated experiment settings.
    :type cfg: ExperimentConfig
    :returns: Accuracy, Wilson bounds, selected C, CV scores, and per-image correctness.
    :rtype: tuple[dict[str, Any], NDArray]
    """
    model, scores = fit_classifier(
        x_train=features[train],
        y_train=labels[train],
        pca_components=PIXEL_PCA_COMPONENTS,
        c_grid=cfg.readout.c_grid,
        cv_folds=cfg.readout.cv_folds,
        seed=cfg.seed,
    )
    with threadpool_limits(limits=1, user_api='blas'):
        correct = model.predict(X=features[evaluation].astype(np.float64)) == labels[evaluation]
    low, high = wilson_interval(successes=int(correct.sum()), n=len(correct))
    return {
        'accuracy': float(correct.mean()),
        'ci_low': low,
        'ci_high': high,
        'ci_method': 'wilson',
        'C': float(model.named_steps['classifier'].C),
        'cv_scores': scores,
    }, correct


def design_check(
    cfg: ExperimentConfig, paths: Paths, split: Literal['val', 'test'] = 'val'
) -> dict[str, Any]:
    """Evaluate five pixel controls and persist a unique reproducible report.

    Validation is the default decision split; test is reserved for frozen protocols.
    Each CV fold fits its own scaler and PCA. The mean-window interval resamples
    images, retaining the dependence between predictions of the same face.

    :param cfg: Experiment settings with balanced splits and episode geometry.
    :type cfg: ExperimentConfig
    :param paths: Experiment storage directories.
    :type paths: Paths
    :param split: Validation or explicitly requested final test partition.
    :type split: Literal['val', 'test']
    :returns: Report identical to the persisted JSON object.
    :rtype: dict[str, Any]
    :raises ValueError: If the evaluation split or image averaging geometry is invalid.
    """
    if split not in {'val', 'test'}:
        raise ValueError('Design check split must be val or test.')
    size = cfg.dataset.preprocess.size
    if size % 2:
        raise ValueError('Whole-image 2 x 2 averaging requires an even aligned image size.')
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    builder = EpisodeBuilder(episodes=cfg.episodes, image_size=size)
    selected = [i for i, sample in enumerate(prepared.samples) if sample.split in {'train', split}]
    samples = [prepared.samples[i] for i in selected]
    train = np.asarray(a=[i for i, sample in enumerate(samples) if sample.split == 'train'])
    evaluation = np.asarray(a=[i for i, sample in enumerate(samples) if sample.split == split])
    labels = np.asarray(a=[sample.label for sample in samples], dtype=np.int64)
    raw_windows = np.stack(
        arrays=[
            builder.build(sample=sample, image=prepared.images[index]).observations
            for index, sample in zip(selected, samples, strict=True)
        ]
    )
    windows = (
        raw_windows.astype(np.float32).reshape(len(samples), cfg.episodes.steps, -1) / PIXEL_MAX
    )
    window_results = []
    correctness = []
    logger = get_logger(name='design_check')
    for step in range(cfg.episodes.steps):
        logger.info('fit_pixel_baseline', baseline='single_window', step=step + 1)
        result, correct = _baseline(
            features=windows[:, step], labels=labels, train=train, evaluation=evaluation, cfg=cfg
        )
        window_results.append(result)
        correctness.append(correct)
    per_image = np.stack(arrays=correctness).mean(axis=0)
    low, high = bootstrap_mean_interval(
        values=per_image,
        samples=cfg.evaluation.bootstrap_samples,
        seed=cfg.evaluation.bootstrap_seed,
    )
    baselines = {
        'first_window': window_results[0],
        'last_window': window_results[-1],
        'single_window_mean': {
            'accuracy': float(per_image.mean()),
            'ci_low': low,
            'ci_high': high,
            'ci_method': 'image_bootstrap',
            'C': [result['C'] for result in window_results],
        },
    }
    logger.info('fit_pixel_baseline', baseline='all_windows')
    baselines['all_windows'], _ = _baseline(
        features=windows.reshape(len(samples), -1),
        labels=labels,
        train=train,
        evaluation=evaluation,
        cfg=cfg,
    )
    images = prepared.images[selected].astype(np.float32) / PIXEL_MAX
    whole = images.reshape(len(samples), size // 2, 2, size // 2, 2, 3).mean(axis=(2, 4))
    logger.info('fit_pixel_baseline', baseline='whole_image')
    baselines['whole_image'], _ = _baseline(
        features=whole.reshape(len(samples), -1),
        labels=labels,
        train=train,
        evaluation=evaluation,
        cfg=cfg,
    )
    gap = 100 * (baselines['all_windows']['accuracy'] - baselines['last_window']['accuracy'])
    warning = gap < cfg.design_check.min_memory_gap_pp
    now = datetime.now(tz=UTC)
    destination = (
        paths.runs / 'design-checks' / f'{now:%Y%m%d-%H%M%S}-{cfg.name}-{uuid4().hex[:8]}.json'
    )
    report = {
        'schema_version': 1,
        'created_utc': now.isoformat(),
        'config_name': cfg.name,
        'config_hash': config_hash(cfg=cfg),
        'dataset_fingerprint': prepared.fingerprint,
        'trajectory_hash': builder.trajectory_hash(
            sample_ids=[s.sample_id for s in prepared.samples]
        ),
        'evaluation_split': split,
        'n_train': len(train),
        'n_eval': len(evaluation),
        'n_classes': len(np.unique(ar=labels)),
        'baselines': baselines,
        'window_baselines': window_results,
        'gap_pp': gap,
        'status': 'warn' if warning else 'ok',
        'message': 'the task barely needs memory' if warning else 'memory gap meets threshold',
        'output': str(destination),
    }
    write_json(path=destination, value=report)
    return report

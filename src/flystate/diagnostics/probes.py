"""Controlled readout probes with training-only selection and persistent evidence."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from flystate.diagnostics.artifacts import attempt, export_classifier, feature_statistics
from flystate.diagnostics.data import load_representation
from flystate.diagnostics.subsets import subset_protocol, training_subset
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import stable_int
from flystate.readouts.fitting import MAX_LOGISTIC_ITERATIONS, fit_classifier
from flystate.readouts.training import accuracy_metrics
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

LABEL_MODES: tuple[str, ...] = ('true', 'permuted', 'memorization')
MEMORIZATION_CLASSES: int = 5
MEMORIZATION_PER_CLASS: int = 4


def score_split(
    model: Pipeline, x: NDArray, targets: NDArray, ids: list[str], directory: Path, split: str
) -> dict[str, Any]:
    """Save class probabilities and score one explicitly named development or in-sample split.

    :param model: Fitted diagnostic classifier.
    :type model: Pipeline
    :param x: Features (N,F).
    :type x: NDArray
    :param targets: Integer target labels (N,).
    :type targets: NDArray
    :param ids: Ordered sample identifiers.
    :type ids: list[str]
    :param directory: Attempt directory.
    :type directory: Path
    :param split: Training, validation, or explicitly in-sample split name.
    :type split: str
    :returns: Top-one/top-five accuracy and natural-log cross entropy.
    :rtype: dict[str, Any]
    """
    probabilities = model.predict_proba(X=x.astype(np.float64))
    classes = model.classes_
    rows = [
        {
            'sample_id': sample_id,
            'y_true': int(target),
            'y_pred': int(classes[np.argmax(a=probability)]),
            'probabilities': probability.tolist(),
        }
        for sample_id, target, probability in zip(ids, targets, probabilities, strict=True)
    ]
    write_table(path=directory / f'{split}-predictions.parquet', rows=rows)
    return {
        **accuracy_metrics(probabilities=probabilities, targets=targets, classes=classes),
        'log_loss': float(log_loss(y_true=targets, y_proba=probabilities, labels=classes)),
    }


def run_probe(
    cfg: ExperimentConfig,
    paths: Paths,
    output: Path,
    representation: str,
    history: str,
    features: str,
    components: int | None,
    label_mode: str,
    max_iterations: int = MAX_LOGISTIC_ITERATIONS,
    train_per_class: int | None = None,
    subset_seed: int = 0,
) -> dict[str, Any]:
    """Run one preregistered diagnostic while retaining failures and all fitted coefficients.

    :param cfg: Original cohort, encoder and trace configuration.
    :type cfg: ExperimentConfig
    :param paths: Working data home.
    :type paths: Paths
    :param output: New attempt directory within that home.
    :type output: Path
    :param representation: Pixels, encoded input, or neural features.
    :type representation: str
    :param history: Last observation or all available observations.
    :type history: str
    :param features: Both neural blocks, spike trace, or voltage.
    :type features: str
    :param components: PCA cap, or null for scaling alone.
    :type components: Optional[int]
    :param label_mode: True training labels, permuted labels, or training-only memorization.
    :type label_mode: str
    :param max_iterations: Explicit solver budget; increasing it requires a new attempt.
    :type max_iterations: int
    :param train_per_class: Optional balanced training support for a nested learning curve.
    :type train_per_class: Optional[int]
    :param subset_seed: Independent nonnegative training-subset seed.
    :type subset_seed: int
    :returns: Completed diagnostic report; no final-test metrics are computed.
    :rtype: dict[str, Any]
    :raises ValueError: If protocol parameters are unsupported.
    """
    parameters = {
        'kind': 'identity_probe',
        'representation': representation,
        'history': history,
        'features': features,
        'pca_components': components,
        'label_mode': label_mode,
        'seed': cfg.seed,
        'tolerance': 1e-6,
        'c_grid': cfg.readout.c_grid,
        'cv_folds': cfg.readout.cv_folds,
        'max_iterations': max_iterations,
    }
    if train_per_class is not None or subset_seed != 0:
        parameters.update(train_per_class=train_per_class, subset_seed=subset_seed)
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        if label_mode not in LABEL_MODES:
            raise ValueError('Unsupported diagnostic label mode.')
        if subset_seed < 0 or (train_per_class is None and subset_seed != 0):
            raise ValueError('A nonnegative subset seed requires explicit training support.')
        if train_per_class is not None and label_mode != 'true':
            raise ValueError('Learning-curve subsets require true labels.')
        data = load_representation(
            cfg=cfg, paths=paths, representation=representation, history=history, features=features
        )
        labels = np.asarray(a=[sample.label for sample in data.samples], dtype=np.int64)
        train = np.flatnonzero(a=[sample.split == 'train' for sample in data.samples])
        val = np.flatnonzero(a=[sample.split == 'val' for sample in data.samples])
        if train_per_class is not None:
            train = training_subset(
                samples=data.samples, per_class=train_per_class, seed=subset_seed
            )
            write_json(
                path=directory / 'training-subset.json',
                value=subset_protocol(
                    samples=data.samples,
                    rows=train,
                    cv_folds=cfg.readout.cv_folds,
                    seed=cfg.seed,
                    components=components,
                    feature_count=data.x.shape[1],
                ),
            )
        if label_mode == 'memorization':
            train = np.concatenate(
                [
                    np.asarray(
                        a=sorted(
                            train[labels[train] == label], key=lambda i: data.samples[i].sample_id
                        )[:MEMORIZATION_PER_CLASS],
                        dtype=np.int64,
                    )
                    for label in np.unique(ar=labels[train])[:MEMORIZATION_CLASSES]
                ]
            )
            val = np.asarray(a=[], dtype=np.int64)
        targets = labels[train].copy()
        if label_mode == 'permuted':
            generator = np.random.default_rng(
                seed=np.random.SeedSequence(entropy=[cfg.seed, stable_int(key='probe-permutation')])
            )
            targets = generator.permutation(x=targets)
        write_json(path=directory / 'provenance.json', value=data.provenance)
        write_table(
            path=directory / 'samples.parquet',
            rows=[
                {**asdict(obj=data.samples[row]), 'fit_label': int(target)}
                for row, target in zip(train, targets, strict=True)
            ]
            + [{**asdict(obj=data.samples[row]), 'fit_label': None} for row in val],
        )
        with threadpool_limits(limits=1, user_api='blas'):
            statistics = feature_statistics(x=data.x[train])
            write_json(path=directory / 'feature-statistics.json', value=statistics)
            model, cv = fit_classifier(
                x_train=data.x[train],
                y_train=targets,
                pca_components=components,
                c_grid=cfg.readout.c_grid,
                cv_folds=cfg.readout.cv_folds,
                seed=cfg.seed,
                tolerance=1e-6,
                max_iterations=max_iterations,
            )
            model_meta = export_classifier(model=model, directory=directory / 'model')
            scores = {}
            for split, rows, truth in (
                ('memorization_train' if label_mode == 'memorization' else 'train', train, targets),
                ('validation', val, labels[val]),
            ):
                if len(rows):
                    scores[split] = score_split(
                        model=model,
                        x=data.x[rows],
                        targets=truth,
                        ids=[data.samples[row].sample_id for row in rows],
                        directory=directory,
                        split=split,
                    )
        write_table(
            path=directory / 'metrics.parquet',
            rows=[{'split': split, **metrics} for split, metrics in scores.items()],
        )
        report = {
            'parameters': parameters,
            'provenance': data.provenance,
            'cv_scores': cv,
            'model': model_meta,
            'classes': model.classes_.tolist(),
            'scores': scores,
            'inference': 'exploratory; final-test and reserve images were not scored',
        }
        write_json(path=directory / 'report.json', value=report)
    return report

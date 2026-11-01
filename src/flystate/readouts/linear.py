"""Portable, validated NumPy inference for fitted scaler/PCA/logistic readouts."""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flystate.experiments.config import ReadoutConfig
from flystate.hashing import sha256_file
from flystate.readouts.fitting import fit_classifier
from flystate.storage.json import write_json
from flystate.storage.runs import assert_mutable

BRAIN_READOUT_TOLERANCE: float = 1e-6
ARRAY_FIELDS: tuple[str, ...] = (
    'scaler_mean',
    'scaler_scale',
    'pca_mean',
    'pca_components',
    'coef',
    'intercept',
    'classes',
)


@dataclass(frozen=True)
class LinearReadout:
    """Immutable float64 inference parameters with sorted integer class identities."""

    scaler_mean: NDArray
    scaler_scale: NDArray
    pca_mean: NDArray
    pca_components: NDArray
    coef: NDArray
    intercept: NDArray
    classes: NDArray
    c_value: float

    def __post_init__(self) -> None:
        """Validate coupled shapes and freeze independent copies of every parameter array.

        :raises ValueError: If model shapes, scales, classes, or values are invalid.
        """
        for field in fields(class_or_instance=self):
            if field.name == 'c_value':
                continue
            original = getattr(self, field.name)
            if field.name == 'classes' and (
                original.dtype.kind not in 'iu' or np.any(original > np.iinfo(np.int64).max)
            ):
                raise ValueError('Readout classes must be int64-compatible integer labels.')
            array = np.array(
                object=original,
                dtype=np.int64 if field.name == 'classes' else np.float64,
                copy=True,
            )
            if not np.isfinite(array).all():
                raise ValueError('Readout parameters must be finite.')
            array.flags.writeable = False
            object.__setattr__(self, field.name, array)
        if self.scaler_mean.ndim != 1 or not self.scaler_mean.size:
            raise ValueError('Scaler mean must be a nonempty feature vector.')
        dimension = len(self.scaler_mean)
        if (
            self.scaler_scale.shape != (dimension,)
            or np.any(self.scaler_scale <= 0)
            or self.pca_mean.shape != (dimension,)
        ):
            raise ValueError('Scaler scales and PCA means must match the feature dimension.')
        if (
            self.pca_components.ndim != 2
            or self.pca_components.shape[1] != dimension
            or not 1 <= self.pca_components.shape[0] <= dimension
        ):
            raise ValueError('PCA components must have shape (K,F) with 1 <= K <= F.')
        if (
            self.classes.ndim != 1
            or len(self.classes) < 2
            or np.any(self.classes[1:] <= self.classes[:-1])
        ):
            raise ValueError('Readout classes must contain at least two sorted distinct labels.')
        rows = 1 if len(self.classes) == 2 else len(self.classes)
        if self.coef.shape != (rows, self.pca_components.shape[0]) or self.intercept.shape != (
            rows,
        ):
            raise ValueError('Logistic coefficient and intercept dimensions are inconsistent.')
        if not np.isfinite(self.c_value) or self.c_value <= 0:
            raise ValueError('Readout C must be finite and positive.')

    def predict_proba(self, x: NDArray) -> NDArray[np.float64]:
        """Apply stable binary or multiclass logistic probabilities without sklearn.

        :param x: Finite numeric features of shape (N,F), converted to float64.
        :type x: NDArray
        :returns: Float64 class probabilities of shape (N,C), in classes order.
        :rtype: NDArray[np.float64]
        :raises ValueError: If input shape or values do not match the fitted model.
        """
        if x.ndim != 2 or x.shape[1] != len(self.scaler_mean) or not np.isfinite(x).all():
            raise ValueError(
                'Prediction features must be finite with the fitted feature dimension.'
            )
        projected = (
            (x.astype(np.float64) - self.scaler_mean) / self.scaler_scale - self.pca_mean
        ) @ self.pca_components.T
        logits = projected @ self.coef.T + self.intercept
        if len(self.classes) == 2:
            positive = np.exp(-np.logaddexp(0, -logits[:, 0]))
            return np.column_stack(tup=(1 - positive, positive))
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)


def fit_readout(
    x_train: NDArray, y_train: NDArray, cfg: ReadoutConfig, seed: int
) -> tuple[LinearReadout, dict[str, float]]:
    """Fit the prescribed fold-local PCA/logistic model and extract portable parameters.

    :param x_train: Training features (N,F).
    :type x_train: NDArray
    :param y_train: Integer training labels (N,).
    :type y_train: NDArray
    :param cfg: Readout component, C-grid, and fold settings.
    :type cfg: ReadoutConfig
    :param seed: Explicit CV and estimator seed.
    :type seed: int
    :returns: Validated portable readout and per-C cross-validation accuracies.
    :rtype: tuple[LinearReadout, dict[str, float]]
    """
    pipeline, scores = fit_classifier(
        x_train=x_train,
        y_train=y_train,
        pca_components=cfg.pca_components,
        c_grid=cfg.c_grid,
        cv_folds=cfg.cv_folds,
        seed=seed,
        tolerance=BRAIN_READOUT_TOLERANCE,
    )
    scaler = pipeline.named_steps['scaler']
    pca = pipeline.named_steps['pca']
    classifier = pipeline.named_steps['classifier']
    return LinearReadout(
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        classes=classifier.classes_,
        c_value=float(classifier.C),
    ), scores


def features_at(features: NDArray, t: int, mode: str) -> NDArray[np.float32]:
    """Select current features or concatenate the available reset history.

    :param features: Numeric episode features (N,T,F).
    :type features: NDArray
    :param t: One-based observation index.
    :type t: int
    :param mode: Persistent, reset, or reset_concat readout policy.
    :type mode: str
    :returns: Float32 readout inputs (N,F) or (N,t*F).
    :rtype: NDArray[np.float32]
    :raises ValueError: If the shape, timestep, or mode is invalid.
    """
    if (
        features.ndim != 3
        or not 1 <= t <= features.shape[1]
        or mode not in {'persistent', 'reset', 'reset_concat'}
    ):
        raise ValueError('Features require shape (N,T,F), a valid timestep, and a supported mode.')
    values = (
        features[:, :t, :].reshape(len(features), t * features.shape[2])
        if mode == 'reset_concat'
        else features[:, t - 1, :]
    )
    return values.astype(np.float32)


def save_readouts(path: Path, readouts: Sequence[LinearReadout], meta: dict[str, Any]) -> None:
    """Publish safe numeric model arrays and metadata once, without overwriting artifacts.

    :param path: Model output directory.
    :type path: Path
    :param readouts: Nonempty ordered per-observation models.
    :type readouts: Sequence[LinearReadout]
    :param meta: Additional metadata, including the readout mode.
    :type meta: dict[str, Any]
    :raises ValueError: If models are absent or target artifacts already exist.
    """
    if (path.parent / 'manifest.json').exists():
        assert_mutable(run_dir=path.parent)
    if not readouts or (path / 'weights.npz').exists() or (path / 'model.json').exists():
        raise ValueError('Models must be nonempty and model artifacts must not already exist.')
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        f't{step:02}_{name}': getattr(model, name)
        for step, model in enumerate(readouts, start=1)
        for name in ARRAY_FIELDS
    }
    with tempfile.NamedTemporaryFile(
        mode='wb', dir=path, suffix='.npz', delete_on_close=False
    ) as stream:
        np.savez(file=stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        Path(stream.name).replace(target=path / 'weights.npz')
    metadata = {
        **meta,
        'kind': 'pca_logistic',
        'timesteps': len(readouts),
        'c_value': {str(t): model.c_value for t, model in enumerate(readouts, start=1)},
        'pca_components': {
            str(t): model.pca_components.shape[0] for t, model in enumerate(readouts, start=1)
        },
        'feature_dim': {
            str(t): len(model.scaler_mean) for t, model in enumerate(readouts, start=1)
        },
        'weights_sha256': sha256_file(path=path / 'weights.npz'),
    }
    write_json(path=path / 'model.json', value=metadata)


def load_readouts(path: Path) -> list[LinearReadout]:
    """Load and validate portable numeric models with pickle disabled.

    :param path: Existing model directory.
    :type path: Path
    :returns: Ordered validated per-observation models.
    :rtype: list[LinearReadout]
    :raises ValueError: If metadata, digest, shapes, or array values are invalid.
    """
    try:
        meta = json.loads(s=(path / 'model.json').read_text(encoding='utf-8'))
        count = meta['timesteps']
        if meta['kind'] != 'pca_logistic' or not isinstance(count, int) or count < 1:
            raise ValueError('Invalid model kind or observation count.')
        if sha256_file(path=path / 'weights.npz') != meta['weights_sha256']:
            raise ValueError('Model weights failed integrity verification.')
        expected = {f't{step:02}_{name}' for step in range(1, count + 1) for name in ARRAY_FIELDS}
        models = []
        with np.load(file=path / 'weights.npz', allow_pickle=False) as arrays:
            if set(arrays.files) != expected:
                raise ValueError('Model array keys do not match the declared timesteps.')
            for step in range(1, count + 1):
                model = LinearReadout(
                    **{name: arrays[f't{step:02}_{name}'] for name in ARRAY_FIELDS},
                    c_value=meta['c_value'][str(step)],
                )
                if (
                    len(model.scaler_mean) != meta['feature_dim'][str(step)]
                    or model.pca_components.shape[0] != meta['pca_components'][str(step)]
                ):
                    raise ValueError('Model metadata dimensions do not match its arrays.')
                models.append(model)
        return models
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Cannot load readouts from {path}: {error}') from error

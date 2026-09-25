"""Attempt-level immutable evidence and safe numeric classifier exports."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file
from flystate.settings import Paths, output_path
from flystate.storage.environment import environment_report, git_state
from flystate.storage.json import write_json

INVENTORY_SEPARATOR: str = '  '
INVENTORY_FILE: str = 'checksums.sha256'
CLASSIFIER_STEP: str = 'classifier'
ITERATIONS: str = 'iterations'
MANIFEST_FILE: str = 'manifest.json'
MODEL_METADATA_FILE: str = 'model.json'
PCA_STEP: str = 'pca'
PCA_EXPLAINED: str = 'pca_explained_variance_ratio'
RSS_HIGH_WATER: str = 'rss_high_water_bytes'
SCALER_STEP: str = 'scaler'
SCHEMA_VERSION_KEY: str = 'schema_version'
STATUS: str = 'status'
ENCODING: str = 'utf-8'
WEIGHTS_FILE: str = 'weights.npz'
WEIGHTS_SHA256: str = 'weights_sha256'


@contextmanager
def attempt(
    paths: Paths, cfg: ExperimentConfig, output: Path, parameters: dict[str, Any]
) -> Iterator[Path]:
    """Record every attempted operation and refuse to overwrite previous evidence.

    :param paths: Working data boundary.
    :type paths: Paths
    :param cfg: Effective source experiment configuration.
    :type cfg: ExperimentConfig
    :param output: New directory inside the data home.
    :type output: Path
    :param parameters: Explicit diagnostic protocol settings.
    :type parameters: dict[str, Any]
    :returns: Context manager yielding the newly reserved output directory.
    :rtype: Iterator[Path]
    :raises Exception: If a diagnostic fails, after preserving its failure manifest.
    """
    directory = output_path(path=output, paths=paths)
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest: dict[str, Any] = {
        SCHEMA_VERSION_KEY: 1,
        STATUS: 'running',
        'created_utc': datetime.now(tz=UTC).isoformat(),
        'parameters': parameters,
        **git_state(),
    }
    write_json(path=directory / 'config.json', value=cfg.model_dump(mode='json'))
    write_json(path=directory / 'environment.json', value=environment_report())
    write_json(path=directory / MANIFEST_FILE, value=manifest)
    try:
        yield directory
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    else:
        manifest[STATUS] = 'completed'
    finally:
        manifest['elapsed_seconds'] = perf_counter() - started
        manifest['rss_end_bytes'] = psutil.Process().memory_info().rss
        status_file = Path('/proc/self/status')
        manifest[RSS_HIGH_WATER] = None
        if status_file.exists():
            for line in status_file.read_text().splitlines():
                if line.startswith('VmHWM:'):
                    manifest[RSS_HIGH_WATER] = int(line.split()[1]) * 1024
        write_json(path=directory / MANIFEST_FILE, value=manifest)
        members = sorted(path for path in directory.rglob('*') if path.is_file())
        inventory = ''.join(
            f'{sha256_file(path=path)}  {path.relative_to(directory)}\n' for path in members
        )
        (directory / INVENTORY_FILE).write_text(data=inventory, encoding=ENCODING)


def verify_attempt_inventory(directory: Path, paths: Paths) -> dict[str, Any]:
    """Verify every recorded file and return the attempt manifest.

    :param directory: Attempt directory within the data home.
    :type directory: Path
    :param paths: Working data boundary.
    :type paths: Paths
    :returns: Stored manifest, including completion or failure status.
    :rtype: dict[str, Any]
    :raises ValueError: If an inventory member is missing, altered, added, or unsafe.
    """
    source = output_path(path=directory, paths=paths)
    inventory = (source / INVENTORY_FILE).read_text(encoding=ENCODING).splitlines()
    names: set[str] = set()
    for line in inventory:
        digest, name = line.split(INVENTORY_SEPARATOR, 1)
        member = (source / name).resolve()
        if name in names or not member.is_relative_to(source) or sha256_file(path=member) != digest:
            raise ValueError('Attempt inventory verification failed.')
        names.add(name)
    actual_names = {
        str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()
    } - {INVENTORY_FILE}
    if not names or names != actual_names:
        raise ValueError('Attempt inventory is incomplete.')
    return json.loads(s=(source / MANIFEST_FILE).read_text(encoding=ENCODING))


def export_classifier(model: Pipeline, directory: Path) -> dict[str, Any]:
    """Preserve scaler, optional PCA, and classifier arrays without pickle or dense identity PCA.

    :param model: Fitted diagnostic scaler/PCA/logistic pipeline.
    :type model: Pipeline
    :param directory: New model output directory.
    :type directory: Path
    :returns: Inference metadata including the learned-array digest.
    :rtype: dict[str, Any]
    """
    directory.mkdir(parents=True, exist_ok=False)
    scaler = model.named_steps[SCALER_STEP]
    classifier = model.named_steps[CLASSIFIER_STEP]
    pca = model.named_steps.get(PCA_STEP)
    arrays = {
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coef': classifier.coef_,
        'intercept': classifier.intercept_,
        'classes': classifier.classes_,
    }
    if pca is not None:
        arrays.update(pca_mean=pca.mean_, pca_components=pca.components_)
    np.savez(file=directory / WEIGHTS_FILE, **arrays)
    metadata = {
        SCHEMA_VERSION_KEY: 1,
        'has_pca': pca is not None,
        PCA_EXPLAINED: [] if pca is None else pca.explained_variance_ratio_.tolist(),
        'C': float(classifier.C),
        ITERATIONS: classifier.n_iter_.tolist(),
        WEIGHTS_SHA256: sha256_file(path=directory / WEIGHTS_FILE),
    }
    write_json(path=directory / MODEL_METADATA_FILE, value=metadata)
    return metadata


def export_linear_readout(model: Pipeline, directory: Path) -> dict[str, Any]:
    """Preserve a scaler/PCA/logistic pipeline as its single equivalent affine map.

    Scaling, PCA and the logistic layer are all affine, so the class scores are
    ``x @ weights.T + bias`` with ``weights = coef @ components / scale`` and
    ``bias = intercept - weights @ mean - coef @ components @ pca_mean``. Storing this
    (K,F) map instead of the (P,F) PCA basis keeps the archive small when F is large. PCA
    diagnostics stay in the metadata.

    :param model: Fitted scaler, PCA and logistic pipeline.
    :type model: Pipeline
    :param directory: New model output directory.
    :type directory: Path
    :returns: Inference metadata including the learned-array digest.
    :rtype: dict[str, Any]
    """
    directory.mkdir(parents=True, exist_ok=False)
    scaler = model.named_steps[SCALER_STEP]
    pca = model.named_steps[PCA_STEP]
    classifier = model.named_steps[CLASSIFIER_STEP]
    projection = classifier.coef_ @ pca.components_
    weights = projection / scaler.scale_
    bias = classifier.intercept_ - weights @ scaler.mean_ - projection @ pca.mean_
    np.savez(file=directory / WEIGHTS_FILE, weights=weights, bias=bias, classes=classifier.classes_)
    metadata = {
        SCHEMA_VERSION_KEY: 1,
        'form': 'affine class scores x @ weights.T + bias, argmax gives classes[index]',
        'pca_components': int(pca.n_components_),
        PCA_EXPLAINED: pca.explained_variance_ratio_.tolist(),
        'C': float(classifier.C),
        ITERATIONS: classifier.n_iter_.tolist(),
        WEIGHTS_SHA256: sha256_file(path=directory / WEIGHTS_FILE),
    }
    write_json(path=directory / MODEL_METADATA_FILE, value=metadata)
    return metadata


def feature_statistics(x: NDArray) -> dict[str, Any]:
    """Describe training feature variability without allocating a feature-by-feature covariance.

    :param x: Finite training matrix (N,F).
    :type x: NDArray
    :returns: Shape, sparsity, variance and centered sample-Gram spectrum diagnostics.
    :rtype: dict[str, Any]
    :raises ValueError: If the matrix is empty, nonfinite, or not two-dimensional.
    """
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Feature statistics require a finite nonempty matrix.')
    values = x.astype(np.float64)
    variances = values.var(axis=0)
    values -= values.mean(axis=0)
    spectrum = np.maximum(np.linalg.eigvalsh(values @ values.T), 0)[::-1]
    total = spectrum.sum()
    fractions = spectrum / total if total > 0 else np.zeros_like(a=spectrum)
    positive = fractions[fractions > 0]
    threshold = np.finfo(np.float64).eps * max(x.shape) * (spectrum[0] if len(spectrum) else 0)
    return {
        'samples': len(x),
        'features': x.shape[1],
        'zero_fraction': float(np.mean(a=x == 0)),
        'near_constant_fraction': float(np.mean(a=variances <= 1e-12)),
        'variance_min': float(variances.min()),
        'variance_median': float(np.median(a=variances)),
        'variance_max': float(variances.max()),
        'numerical_rank': int(np.count_nonzero(spectrum > threshold)),
        'effective_rank': float(np.exp(-np.sum(positive * np.log(positive)))) if total > 0 else 0.0,
        'centered_spectrum_fraction': fractions.tolist(),
    }

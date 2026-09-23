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
        'schema_version': 1,
        'status': 'running',
        'created_utc': datetime.now(tz=UTC).isoformat(),
        'parameters': parameters,
        **git_state(),
    }
    write_json(path=directory / 'config.json', value=cfg.model_dump(mode='json'))
    write_json(path=directory / 'environment.json', value=environment_report())
    write_json(path=directory / 'manifest.json', value=manifest)
    try:
        yield directory
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    else:
        manifest['status'] = 'completed'
    finally:
        manifest['elapsed_seconds'] = perf_counter() - started
        manifest['rss_end_bytes'] = psutil.Process().memory_info().rss
        status_file = Path('/proc/self/status')
        manifest['rss_high_water_bytes'] = None
        if status_file.exists():
            for line in status_file.read_text().splitlines():
                if line.startswith('VmHWM:'):
                    manifest['rss_high_water_bytes'] = int(line.split()[1]) * 1024
        write_json(path=directory / 'manifest.json', value=manifest)
        members = sorted(path for path in directory.rglob('*') if path.is_file())
        inventory = ''.join(
            f'{sha256_file(path=path)}  {path.relative_to(directory)}\n' for path in members
        )
        (directory / 'checksums.sha256').write_text(data=inventory, encoding='utf-8')


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
    inventory = (source / 'checksums.sha256').read_text(encoding='utf-8').splitlines()
    names: set[str] = set()
    for line in inventory:
        digest, name = line.split('  ', 1)
        member = (source / name).resolve()
        if name in names or not member.is_relative_to(source) or sha256_file(path=member) != digest:
            raise ValueError('Attempt inventory verification failed.')
        names.add(name)
    actual_names = {
        str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()
    } - {'checksums.sha256'}
    if not names or names != actual_names:
        raise ValueError('Attempt inventory is incomplete.')
    return json.loads(s=(source / 'manifest.json').read_text(encoding='utf-8'))


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
    scaler = model.named_steps['scaler']
    classifier = model.named_steps['classifier']
    pca = model.named_steps.get('pca')
    arrays = {
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coef': classifier.coef_,
        'intercept': classifier.intercept_,
        'classes': classifier.classes_,
    }
    if pca is not None:
        arrays.update(pca_mean=pca.mean_, pca_components=pca.components_)
    np.savez(file=directory / 'weights.npz', **arrays)
    metadata = {
        'schema_version': 1,
        'has_pca': pca is not None,
        'pca_explained_variance_ratio': []
        if pca is None
        else pca.explained_variance_ratio_.tolist(),
        'C': float(classifier.C),
        'iterations': classifier.n_iter_.tolist(),
        'weights_sha256': sha256_file(path=directory / 'weights.npz'),
    }
    write_json(path=directory / 'model.json', value=metadata)
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

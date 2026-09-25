"""Verify the T35 coefficient backfill against the immutable confirmation result."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.input_access import _features
from flystate.experiments.config import load_config
from flystate.settings import get_paths

STUDY: Path = Path(__file__).resolve().parents[1]
EVALUATION: str = 'evaluate'
EXPORT: str = 'evaluate-model-export'
PREDICTIONS: str = 'predictions.parquet'
REPORT: str = 'report.json'
ARRAYS: str = 'weights.npz'
METADATA: str = 'model.json'
RECORDINGS: tuple[str, ...] = ('persistent', 'reset', 'shuffled', 'shuffled-reset')


def _sha256(path: Path) -> str:
    """Hash a file without loading the entire payload into memory.

    :param path: File to hash.
    :type path: Path
    :returns: Lowercase SHA-256 hex digest.
    :rtype: str
    """
    digest = hashlib.sha256()
    with path.open(mode='rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from an archived file.

    :param path: JSON file to read.
    :type path: Path
    :returns: Parsed JSON object.
    :rtype: dict[str, Any]
    :raises ValueError: If the file does not contain an object.
    """
    value = json.loads(s=path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _verify_models(export: Path, scores: dict[str, Any]) -> dict[str, int]:
    """Check every coefficient array and its agreement with the evaluation report.

    :param export: Archived coefficient-export attempt.
    :type export: Path
    :param scores: Per-case score and model metadata mapping.
    :type scores: dict[str, Any]
    :returns: Verified model count and combined array bytes.
    :rtype: dict[str, int]
    :raises ValueError: If a model is missing, changed, nonfinite or inconsistent.
    """
    total_bytes = 0
    for case, summary in scores.items():
        directory = export / 'models' / case
        weights = directory / ARRAYS
        metadata = _read_json(path=directory / METADATA)
        if metadata != summary['model'] or metadata['C'] != summary['C']:
            raise ValueError(f'Model metadata differs from the report: {case}')
        if _sha256(path=weights) != metadata['weights_sha256']:
            raise ValueError(f'Model array digest differs: {case}')
        with np.load(file=weights, allow_pickle=False) as arrays:
            required = {'scaler_mean', 'scaler_scale', 'coef', 'intercept', 'classes'}
            if not required <= set(arrays.files):
                raise ValueError(f'Model arrays are incomplete: {case}')
            if not all(np.isfinite(arrays[name]).all() for name in arrays.files):
                raise ValueError(f'Model arrays contain nonfinite values: {case}')
            features = arrays['scaler_mean'].shape[0]
            coefficients = arrays['coef']
            if (
                arrays['scaler_scale'].shape != (features,)
                or arrays['classes'].shape != (coefficients.shape[0],)
                or arrays['intercept'].shape != (coefficients.shape[0],)
                or arrays['pca_mean'].shape != (features,)
                or arrays['pca_components'].shape != (coefficients.shape[1], features)
            ):
                raise ValueError(f'Model array shapes are inconsistent: {case}')
        total_bytes += weights.stat().st_size
    actual = list((export / 'models').rglob(ARRAYS))
    if len(actual) != len(scores):
        raise ValueError('The model directory contains an unexpected number of arrays.')
    return {'models': len(scores), 'model_bytes': total_bytes}


def _predict_from_arrays(weights: Path, features: NDArray[np.float32]) -> NDArray[np.int64]:
    """Apply archived scaler, PCA and logistic coefficients without sklearn inference.

    :param weights: Numeric arrays exported by the fitted classifier.
    :type weights: Path
    :param features: Held-out features of shape (M,F), float32.
    :type features: NDArray[np.float32]
    :returns: Predicted identity labels of shape (M,), int64.
    :rtype: NDArray[np.int64]
    """
    with np.load(file=weights, allow_pickle=False) as arrays:
        values = (features.astype(np.float64) - arrays['scaler_mean']) / arrays['scaler_scale']
        projected = (values - arrays['pca_mean']) @ arrays['pca_components'].T
        logits = projected @ arrays['coef'].T + arrays['intercept']
        return arrays['classes'][np.argmax(a=logits, axis=1)].astype(np.int64)


def _verify_inference(study: Path, home: Path, expected_rows: list[dict[str, Any]]) -> int:
    """Replay all archived labels from the exported arrays and original feature sources.

    :param study: T35 study directory with model arrays.
    :type study: Path
    :param home: Registered external dataset and original recording home.
    :type home: Path
    :param expected_rows: Frozen original per-photograph prediction rows.
    :type expected_rows: list[dict[str, Any]]
    :returns: Number of exactly reproduced predictions.
    :rtype: int
    :raises ValueError: If cohort order or any predicted identity differs.
    """
    paths = get_paths(home=home)
    cfg = load_config(path=study.parents[2] / 'configs/celeba-confirm.yaml')
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    samples = prepared.samples
    held_out = np.asarray([sample.split != 'train' for sample in samples], dtype=np.bool_)
    sample_ids = [sample.sample_id for sample, keep in zip(samples, held_out, strict=True) if keep]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in expected_rows:
        by_case.setdefault(row['case'], []).append(row)
    references, _ = _features(
        cfg=cfg, paths=paths, prepared=prepared, rows=list(range(len(samples)))
    )
    width = references['encoded_current'].shape[1] // cfg.episodes.steps
    cases = {
        'input/encoded_current_all': references['encoded_current'],
        'input/encoded_current_last': references['encoded_current'][:, -width:],
        'input/pixels_all': references['pixels'],
    }
    export = study / 'snapshot' / EXPORT
    replayed = 0
    with threadpool_limits(limits=1, user_api='blas'):
        for recording in RECORDINGS:
            source = (
                home
                / 'runs/diagnostics/2026-09-25-confirmation/record'
                / recording
                / 'responses.npz'
            )
            with np.load(file=source, allow_pickle=False) as arrays:
                for case in by_case:
                    if case.startswith(f'{recording}/'):
                        population = case.split('/', 1)[1]
                        cases[case] = arrays[f'final_{population}']
        for case, features in cases.items():
            expected = by_case[case]
            if [row['sample_id'] for row in expected] != sample_ids:
                raise ValueError(f'Held-out sample order differs: {case}')
            weights = export / 'models' / case / ARRAYS
            predicted = _predict_from_arrays(weights=weights, features=features[held_out])
            if predicted.tolist() != [row['predicted'] for row in expected]:
                raise ValueError(f'Exported classifier does not reproduce predictions: {case}')
            replayed += len(expected)
    if replayed != len(expected_rows):
        raise ValueError('Not every frozen prediction was replayed.')
    return replayed


def verify(study: Path, home: Path | None) -> dict[str, Any]:
    """Compare the model-export attempt with the immutable original confirmation.

    :param study: T35 archived study directory.
    :type study: Path
    :param home: Optional data home for full inference replay.
    :type home: Optional[Path]
    :returns: Counts and digests of matching predictions and models.
    :rtype: dict[str, Any]
    :raises ValueError: If results, model files or optional inference disagree.
    """
    original = study / 'snapshot' / EVALUATION
    export = study / 'snapshot' / EXPORT
    original_predictions = original / PREDICTIONS
    exported_predictions = export / PREDICTIONS
    original_digest = _sha256(path=original_predictions)
    if original_digest != _sha256(path=exported_predictions):
        raise ValueError('The frozen prediction file differs from the model-export replay.')
    original_report = _read_json(path=original / REPORT)
    exported_report = _read_json(path=export / REPORT)
    stripped_scores = {
        case: {key: value for key, value in score.items() if key != 'model'}
        for case, score in exported_report['scores'].items()
    }
    if {**exported_report, 'scores': stripped_scores} != original_report:
        raise ValueError('Frozen scores or fitting choices differ from the model-export replay.')
    models = _verify_models(export=export, scores=exported_report['scores'])
    rows = pq.read_table(source=original_predictions).to_pylist()
    result: dict[str, Any] = {
        **models,
        'prediction_rows': len(rows),
        'predictions_sha256': original_digest,
        'scores_exact': True,
    }
    if home is not None:
        result['inference_rows_exact'] = _verify_inference(
            study=study, home=home, expected_rows=rows
        )
    return result


def main() -> None:
    """Print a single JSON verification result for the archived model export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, help='Data home for full numeric inference replay.')
    options = parser.parse_args()
    result = verify(study=STUDY, home=options.home)
    sys.stdout.write(json.dumps(result, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

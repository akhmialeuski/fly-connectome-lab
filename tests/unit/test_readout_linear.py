"""Verify portable logistic inference, strict model validation, and safe persistence."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from flystate.experiments.config import ReadoutConfig
from flystate.hashing import sha256_file
from flystate.readouts.fitting import fit_classifier
from flystate.readouts.linear import (
    LinearReadout,
    features_at,
    fit_readout,
    load_readouts,
    save_readouts,
)
from flystate.storage.json import write_json


@pytest.fixture
def binary_model() -> LinearReadout:
    """Provide simple valid parameters with two features and two classes.

    :returns: Immutable binary model.
    :rtype: LinearReadout
    """
    return LinearReadout(
        scaler_mean=np.zeros(shape=2),
        scaler_scale=np.ones(shape=2),
        pca_mean=np.zeros(shape=2),
        pca_components=np.eye(N=2),
        coef=np.ones(shape=(1, 2)),
        intercept=np.zeros(shape=1),
        classes=np.asarray(a=[0, 1]),
        c_value=1.0,
    )


class TestLinearReadout:
    """Compare mathematical predictions with fitted sklearn models and durable numeric artifacts."""

    @pytest.mark.parametrize('classes', [2, 5])
    def test_sklearn_parity(self, classes: int, tmp_path: Path) -> None:
        """Fit seeded models, compare probabilities, then verify exact persistence round trips.

        :param classes: Binary or multiclass classification size.
        :type classes: int
        :param tmp_path: Isolated model artifact directory.
        :type tmp_path: Path
        """
        generator = np.random.default_rng(seed=4)
        x = generator.normal(size=(classes * 12, 8))
        x[:, -1] = 3
        y = np.repeat(a=np.arange(classes), repeats=12)
        cfg = ReadoutConfig(pca_components=5, c_grid=(0.01, 0.1), cv_folds=3)
        reference, scores = fit_classifier(
            x_train=x,
            y_train=y,
            pca_components=5,
            c_grid=cfg.c_grid,
            cv_folds=3,
            seed=0,
            tolerance=1e-6,
        )
        model, actual_scores = fit_readout(x_train=x, y_train=y, cfg=cfg, seed=0)
        test = generator.normal(size=(10, 8))
        np.testing.assert_allclose(
            actual=model.predict_proba(x=test), desired=reference.predict_proba(X=test), atol=1e-6
        )
        assert actual_scores == scores and model.scaler_scale[-1] == 1
        assert not model.coef.flags.writeable
        save_readouts(path=tmp_path, readouts=[model, model], meta={'mode': 'persistent'})
        loaded = load_readouts(path=tmp_path)
        np.testing.assert_array_equal(
            actual=loaded[1].predict_proba(x=test), desired=model.predict_proba(x=test)
        )
        with pytest.raises(expected_exception=ValueError, match='already exist'):
            save_readouts(path=tmp_path, readouts=[model], meta={})

    def test_extreme_logits(self, binary_model: LinearReadout) -> None:
        """Keep binary sigmoid probabilities finite at very large positive and negative logits.

        :param binary_model: Simple validated binary model.
        :type binary_model: LinearReadout
        """
        result = binary_model.predict_proba(x=np.asarray(a=[[1e6, 1e6], [-1e6, -1e6]]))
        np.testing.assert_array_equal(actual=result, desired=[[0, 1], [1, 0]])

    @pytest.mark.parametrize(
        'field,value',
        [
            ('classes', np.asarray([0.0, 1.0])),
            ('classes', np.asarray([0, 2**64 - 1], dtype=np.uint64)),
            ('scaler_mean', np.asarray([np.nan, 0])),
            ('scaler_mean', np.zeros((1, 2))),
            ('scaler_scale', np.asarray([0.0, 1.0])),
            ('pca_mean', np.zeros(3)),
            ('pca_components', np.zeros((0, 2))),
            ('classes', np.asarray([1, 1])),
            ('coef', np.zeros((2, 2))),
            ('intercept', np.zeros(2)),
            ('c_value', 0.0),
        ],
    )
    def test_invalid_model(
        self, binary_model: LinearReadout, field: str, value: NDArray | float
    ) -> None:
        """Reject inconsistent parameters before prediction or model publication.

        :param binary_model: Valid parameter template.
        :type binary_model: LinearReadout
        :param field: Parameter to invalidate.
        :type field: str
        :param value: Invalid parameter value.
        :type value: NDArray | float
        """
        changes: dict[str, Any] = {field: value}
        with pytest.raises(expected_exception=ValueError):
            replace(binary_model, **changes)

    @pytest.mark.parametrize('x', [np.zeros(2), np.zeros((1, 3)), np.full((1, 2), np.nan)])
    def test_invalid_prediction(self, binary_model: LinearReadout, x: NDArray) -> None:
        """Reject malformed or nonfinite input features.

        :param binary_model: Valid model.
        :type binary_model: LinearReadout
        :param x: Invalid feature matrix.
        :type x: NDArray
        """
        with pytest.raises(expected_exception=ValueError):
            binary_model.predict_proba(x=x)

    @pytest.mark.parametrize('bad', ['kind', 'count', 'digest', 'keys', 'dimensions', 'missing'])
    def test_invalid_artifacts(self, binary_model: LinearReadout, tmp_path: Path, bad: str) -> None:
        """Detect corrupted metadata and weights, including an updated digest over wrong arrays.

        :param binary_model: Valid model to serialize.
        :type binary_model: LinearReadout
        :param tmp_path: Isolated model directory.
        :type tmp_path: Path
        :param bad: Corruption category.
        :type bad: str
        """
        save_readouts(path=tmp_path, readouts=[binary_model], meta={})
        meta = json.loads(s=(tmp_path / 'model.json').read_text())
        if bad == 'kind':
            meta['kind'] = 'unsupported'
        elif bad == 'count':
            meta['timesteps'] = 0
        elif bad == 'digest':
            meta['weights_sha256'] = 'wrong'
        elif bad == 'keys':
            np.savez(file=tmp_path / 'weights.npz', unknown=np.zeros(1))
            meta['weights_sha256'] = sha256_file(path=tmp_path / 'weights.npz')
        elif bad == 'dimensions':
            meta['feature_dim']['1'] = 3
        else:
            (tmp_path / 'weights.npz').unlink()
        write_json(path=tmp_path / 'model.json', value=meta)
        with pytest.raises(expected_exception=ValueError):
            load_readouts(path=tmp_path)

    def test_failed_save(self, binary_model: LinearReadout, tmp_path: Path) -> None:
        """Reject empty model lists and clean temporary files after interrupted array writes.

        :param binary_model: Valid model.
        :type binary_model: LinearReadout
        :param tmp_path: Isolated model directory.
        :type tmp_path: Path
        """
        with pytest.raises(expected_exception=ValueError):
            save_readouts(path=tmp_path, readouts=[], meta={})
        with (
            patch(target='flystate.readouts.linear.np.savez', side_effect=KeyboardInterrupt),
            pytest.raises(expected_exception=KeyboardInterrupt),
        ):
            save_readouts(path=tmp_path, readouts=[binary_model], meta={})
        assert list(tmp_path.iterdir()) == []
        with (
            patch(
                target='flystate.readouts.linear.tempfile.NamedTemporaryFile',
                side_effect=OSError('disk full'),
            ),
            pytest.raises(expected_exception=OSError, match='disk full'),
        ):
            save_readouts(path=tmp_path, readouts=[binary_model], meta={})
        assert list(tmp_path.iterdir()) == []

    def test_feature_selection(self) -> None:
        """Select current rows or exact chronological concatenation and reject invalid requests."""
        features = np.arange(24).reshape(2, 3, 4)
        for mode in ('persistent', 'reset'):
            np.testing.assert_array_equal(
                actual=features_at(features=features, t=2, mode=mode), desired=features[:, 1]
            )
        values = features_at(features=features, t=2, mode='reset_concat')
        assert values.shape == (2, 8) and values.dtype == np.float32
        np.testing.assert_array_equal(actual=values, desired=features[:, :2].reshape(2, 8))
        for t, mode in ((0, 'reset'), (4, 'reset'), (1, 'invalid')):
            with pytest.raises(expected_exception=ValueError):
                features_at(features=features, t=t, mode=mode)
        with pytest.raises(expected_exception=ValueError):
            features_at(features=features[0], t=1, mode='reset')
        with pytest.raises(expected_exception=ValueError):
            fit_classifier(
                x_train=np.ones((4, 2)),
                y_train=np.asarray([0, 0, 1, 1]),
                pca_components=1,
                c_grid=[1.0],
                cv_folds=2,
                seed=0,
                tolerance=0,
            )

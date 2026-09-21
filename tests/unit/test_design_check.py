"""Verify fold isolation, image-level intervals, and pixel control reports."""

import json
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets import registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.evaluation.design_check import design_check
from flystate.evaluation.stats import bootstrap_mean_interval, wilson_interval
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.readouts import fitting
from flystate.settings import get_paths


@pytest.fixture
def pixel_config(synthetic_celeba_dir: Path) -> ExperimentConfig:
    """Register synthetic images and configure inexpensive real classifier fits.

    :param synthetic_celeba_dir: Generated CelebA-shaped source.
    :type synthetic_celeba_dir: Path
    :returns: Four-class, two-window experiment.
    :rtype: ExperimentConfig
    """
    paths = get_paths()
    registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
    registry.set_validation(
        paths=paths,
        name='celeba',
        full=True,
        report=CelebAAdapter(root=synthetic_celeba_dir, expected=None).validate(full=True),
    )
    return ExperimentConfig.model_validate(
        obj={
            'name': 'pixels',
            'dataset': {'subset': {'n_identities': 4}},
            'episodes': {'steps': 2, 'window': 8},
            'readout': {'c_grid': [0.01, 0.1], 'cv_folds': 2},
            'evaluation': {'bootstrap_samples': 100},
        }
    )


class TestStatistics:
    """Check textbook intervals and deterministic independent-image resampling."""

    def test_intervals(self) -> None:
        """Match Wilson reference values and bootstrap endpoints for constant observations."""
        assert wilson_interval(successes=8, n=10) == pytest.approx((0.49016, 0.94332), abs=0.00001)
        values = np.asarray(a=[0.0, 0.5, 1.0])
        first = bootstrap_mean_interval(values=values, samples=513, seed=42)
        assert first == bootstrap_mean_interval(values=values, samples=513, seed=42)
        assert first[0] <= values.mean() <= first[1]
        assert bootstrap_mean_interval(values=np.ones(shape=10), samples=100, seed=0) == (1, 1)

    @pytest.mark.parametrize(
        'successes,n,confidence', [(-1, 10, 0.95), (11, 10, 0.95), (0, 0, 0.95), (1, 2, 1)]
    )
    def test_invalid_wilson(self, successes: int, n: int, confidence: float) -> None:
        """Reject invalid counts and confidence.

        :param successes: Success count.
        :type successes: int
        :param n: Observation count.
        :type n: int
        :param confidence: Requested confidence level.
        :type confidence: float
        """
        with pytest.raises(expected_exception=ValueError):
            wilson_interval(successes=successes, n=n, confidence=confidence)

    @pytest.mark.parametrize(
        'values,samples,seed',
        [([], 10, 0), ([[1]], 10, 0), ([float('nan')], 10, 0), ([1], 0, 0), ([1], 10, -1)],
    )
    def test_invalid_bootstrap(self, values: list, samples: int, seed: int) -> None:
        """Reject malformed independent units and invalid resampling settings.

        :param values: Invalid observations or valid control input.
        :type values: list
        :param samples: Resample count.
        :type samples: int
        :param seed: Resampling seed.
        :type seed: int
        """
        with pytest.raises(expected_exception=ValueError):
            bootstrap_mean_interval(values=np.asarray(a=values), samples=samples, seed=seed)


class TestClassifier:
    """Verify fold-local preprocessing, deterministic C selection, and input guards."""

    @pytest.mark.parametrize('components', [100, None])
    def test_fold_isolation(self, components: int | None) -> None:
        """Record projection inputs, fit a classifier, then match every fold and final refit.

        :param components: PCA cap or scaling without PCA.
        :type components: Optional[int]
        """
        x = np.random.default_rng(seed=0).normal(size=(24, 5))
        y = np.repeat(a=[0, 1], repeats=12)
        x[:, 0] += y * 10
        with patch.object(
            target=fitting, attribute='_projection', wraps=fitting._projection
        ) as spy:
            model, scores = fitting.fit_classifier(
                x_train=x,
                y_train=y,
                pca_components=components,
                c_grid=[1.0, 0.1],
                cv_folds=3,
                seed=0,
            )
        splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(X=x, y=y))
        assert spy.call_count == 4
        for call, (train, _) in zip(spy.call_args_list[:-1], splits, strict=True):
            np.testing.assert_array_equal(actual=call.kwargs['x'], desired=x[train])
        np.testing.assert_array_equal(actual=spy.call_args_list[-1].kwargs['x'], desired=x)
        assert model.score(X=x, y=y) == 1
        assert scores == {'0.1': 1.0, '1.0': 1.0}
        assert model.named_steps['classifier'].C == 0.1
        assert model.named_steps['classifier'].tol == 1e-4

    def test_nonconvergence(self) -> None:
        """Surface optimizer failure instead of publishing an unconverged result."""
        x = np.random.default_rng(seed=1).normal(size=(40, 10))
        y = np.repeat(a=[0, 1], repeats=20)
        with (
            patch.object(target=fitting, attribute='MAX_LOGISTIC_ITERATIONS', new=1),
            pytest.raises(expected_exception=ConvergenceWarning),
        ):
            fitting.fit_classifier(
                x_train=x, y_train=y, pca_components=5, c_grid=[1.0], cv_folds=2, seed=0
            )

    @pytest.mark.parametrize(
        'case',
        [
            'dimensions',
            'empty',
            'nan',
            'labels',
            'float_labels',
            'classes',
            'support',
            'pca',
            'folds',
            'grid',
            'negative_c',
            'infinite_c',
        ],
    )
    def test_invalid_inputs(self, case: str) -> None:
        """Reject scientifically invalid training inputs before fitting.

        :param case: Invalid data or hyperparameter category.
        :type case: str
        """
        x: NDArray = np.ones(shape=(4, 2))
        y = np.asarray(a=[0, 0, 1, 1])
        if case == 'dimensions':
            x = x.ravel()
        elif case == 'empty':
            x = x[:, :0]
        elif case == 'nan':
            x[0, 0] = np.nan
        elif case == 'labels':
            y = y[:2]
        elif case == 'float_labels':
            y = y.astype(np.float64)
        elif case == 'classes':
            y[:] = 0
        elif case == 'support':
            y[-1] = 2
        with pytest.raises(expected_exception=ValueError):
            fitting.fit_classifier(
                x_train=x,
                y_train=y,
                pca_components=0 if case == 'pca' else 2,
                cv_folds=1 if case == 'folds' else 2,
                c_grid=[]
                if case == 'grid'
                else [-1]
                if case == 'negative_c'
                else [np.inf]
                if case == 'infinite_c'
                else [1],
                seed=0,
            )


class TestDesignCheck:
    """Exercise actual image preparation, classifiers, persisted reports, and CLI errors."""

    def test_cli(self, pixel_config: ExperimentConfig, tmp_path: Path) -> None:
        """Write a small configuration, run both output modes, and verify all five controls.

        :param pixel_config: Registered synthetic experiment.
        :type pixel_config: ExperimentConfig
        :param tmp_path: Isolated configuration directory.
        :type tmp_path: Path
        """
        path = tmp_path / 'pixels.yaml'
        path.write_text(data=effective_yaml(cfg=pixel_config))
        runner = CliRunner()
        result = runner.invoke(app=app, args=['dataset', 'design-check', str(path), '--json'])
        assert result.exit_code == 0, result.output
        report = json.loads(s=result.stdout)
        assert json.loads(s=Path(report['output']).read_text()) == report
        assert report['evaluation_split'] == 'val'
        assert report['n_train'] == 56 and report['n_eval'] == 12
        assert set(report['baselines']) == {
            'first_window',
            'last_window',
            'single_window_mean',
            'all_windows',
            'whole_image',
        }
        assert report['baselines']['single_window_mean']['ci_method'] == 'image_bootstrap'
        assert len(report['baselines']['single_window_mean']['C']) == 2
        with patch(target='flystate.cli.dataset.design_check', return_value=report):
            text = runner.invoke(app=app, args=['dataset', 'design-check', str(path)])
        assert text.exit_code == 0 and '95% interval' in text.stdout
        missing = runner.invoke(
            app=app, args=['dataset', 'design-check', str(tmp_path / 'missing'), '--json']
        )
        assert missing.exit_code == 2 and 'error' in json.loads(s=missing.stdout)
        with patch(
            target='flystate.cli.dataset.design_check', side_effect=ValueError('invalid pixels')
        ):
            failed = runner.invoke(app=app, args=['dataset', 'design-check', str(path), '--json'])
        assert failed.exit_code == 1 and json.loads(s=failed.stdout)['error'] == 'invalid pixels'

    def test_threshold_and_test_split(self, pixel_config: ExperimentConfig) -> None:
        """Control classifier outputs to verify the gap threshold and explicit final-test reporting.

        :param pixel_config: Registered synthetic experiment.
        :type pixel_config: ExperimentConfig
        """
        outcomes = [
            ({'accuracy': value, 'C': 1}, np.ones(shape=12, dtype=bool))
            for value in [0.4, 0.5, 0.55, 0.7]
        ]
        with patch(target='flystate.evaluation.design_check._baseline', side_effect=outcomes):
            report = design_check(cfg=pixel_config, paths=get_paths(), split='test')
        assert report['gap_pp'] == pytest.approx(5)
        assert report['status'] == 'warn' and report['message'] == 'the task barely needs memory'
        assert report['evaluation_split'] == 'test'
        outcomes[2][0]['accuracy'] = 0.8
        with patch(target='flystate.evaluation.design_check._baseline', side_effect=outcomes):
            report = design_check(cfg=pixel_config, paths=get_paths())
        assert report['status'] == 'ok'

    def test_invalid_split(self) -> None:
        """Reject an unsupported partition at the library boundary."""
        with pytest.raises(expected_exception=ValueError, match='split'):
            design_check(
                cfg=ExperimentConfig(name='invalid'),
                paths=get_paths(),
                split=cast(Literal['val', 'test'], 'train'),
            )

    def test_odd_size(self) -> None:
        """Reject averaging geometry that cannot represent complete two-pixel blocks."""
        cfg = ExperimentConfig.model_validate(
            obj={'name': 'odd', 'dataset': {'preprocess': {'size': 127}}}
        )
        with pytest.raises(expected_exception=ValueError, match='even'):
            design_check(cfg=cfg, paths=get_paths())

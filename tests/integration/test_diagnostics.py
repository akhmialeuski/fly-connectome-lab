"""Offline evidence integrity, split isolation, and diagnostic classifier replay."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, softmax
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics.artifacts import attempt, export_classifier, feature_statistics
from flystate.diagnostics.audit import audit_cohort, duplicate_kind
from flystate.diagnostics.convergence import diagnose_convergence
from flystate.diagnostics.data import load_representation
from flystate.diagnostics.probes import run_probe
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_file
from flystate.readouts.fitting import fit_classifier
from flystate.settings import get_paths
from flystate.storage.parquet import read_table
from flystate.traces.builder import build_trace


class TestDiagnosticArtifacts:
    """Validate numeric replay, matrix diagnostics, and immutable failure evidence."""

    @pytest.mark.parametrize('components', [None, 3])
    @pytest.mark.parametrize('class_count', [2, 4])
    def test_export_replay(self, tmp_path: Path, components: int | None, class_count: int) -> None:
        """Fit a classifier, export numeric arrays, and independently replay its probabilities.

        :param tmp_path: Isolated model directory.
        :type tmp_path: Path
        :param components: Optional PCA cap.
        :type components: Optional[int]
        :param class_count: Binary or multiclass problem size.
        :type class_count: int
        """
        generator = np.random.default_rng(seed=0)
        x = generator.normal(size=(40, 8))
        y = np.arange(40) % class_count
        model, _ = fit_classifier(
            x_train=x,
            y_train=y,
            pca_components=components,
            c_grid=[0.1, 1.0],
            cv_folds=2,
            seed=0,
        )
        directory = tmp_path / 'model'
        meta = export_classifier(model=model, directory=directory)
        assert meta['weights_sha256'] == sha256_file(path=directory / 'weights.npz')
        with np.load(file=directory / 'weights.npz', allow_pickle=False) as arrays:
            z = (x - arrays['scaler_mean']) / arrays['scaler_scale']
            if components is not None:
                z = (z - arrays['pca_mean']) @ arrays['pca_components'].T
            else:
                assert 'pca_components' not in arrays.files
            logits = z @ arrays['coef'].T + arrays['intercept']
            probability = (
                np.column_stack((1 - expit(logits[:, 0]), expit(logits[:, 0])))
                if class_count == 2
                else softmax(logits, axis=1)
            )
        np.testing.assert_allclose(actual=probability, desired=model.predict_proba(X=x), atol=1e-12)

    def test_statistics(self) -> None:
        """Verify constant and rank-one spectra, then reject invalid matrices."""
        assert feature_statistics(x=np.zeros(shape=(4, 3)))['effective_rank'] == 0
        rank_one = feature_statistics(x=np.arange(4)[:, None] * np.ones(shape=(1, 3)))
        assert rank_one['numerical_rank'] == 1
        assert rank_one['effective_rank'] == pytest.approx(expected=1)
        for invalid in (np.zeros(shape=(0, 3)), np.array([1]), np.array([[np.nan]])):
            with pytest.raises(expected_exception=ValueError):
                feature_statistics(x=invalid)

    def test_failed_attempt(self, tiny_experiment: ExperimentConfig) -> None:
        """Force a failed attempt, verify its checksums, then reject overwriting its evidence.

        :param tiny_experiment: Offline source configuration.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        output = paths.runs / 'failure'
        with (
            pytest.raises(expected_exception=RuntimeError, match='intentional'),
            attempt(paths=paths, cfg=tiny_experiment, output=output, parameters={}),
        ):
            raise RuntimeError('intentional')
        manifest = json.loads(s=(output / 'manifest.json').read_text())
        assert manifest['status'] == 'failed' and manifest['error_type'] == 'RuntimeError'
        for line in (output / 'checksums.sha256').read_text().splitlines():
            digest, relative = line.split('  ', 1)
            assert sha256_file(path=output / relative) == digest
        with (
            pytest.raises(expected_exception=FileExistsError),
            attempt(paths=paths, cfg=tiny_experiment, output=output, parameters={}),
        ):
            pytest.fail(reason='A completed or failed directory must never be reused.')


class TestDevelopmentProbes:
    """Exercise all representation sources without exposing final-test rows to fitting."""

    def test_representations(self, tiny_experiment: ExperimentConfig) -> None:
        """Build synthetic traces, compare feature blocks, then verify source and row identities.

        :param tiny_experiment: Offline source experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        build_trace(cfg=tiny_experiment, paths=paths)
        blocks = {}
        for features in ('both', 'spike_trace', 'voltage'):
            blocks[features] = load_representation(
                cfg=tiny_experiment,
                paths=paths,
                representation='neural',
                history='last',
                features=features,
            )
        np.testing.assert_array_equal(
            actual=blocks['both'].x,
            desired=np.concatenate((blocks['spike_trace'].x, blocks['voltage'].x), axis=1),
        )
        for representation in ('pixels', 'encoded', 'neural'):
            for history in ('last', 'all'):
                data = load_representation(
                    cfg=tiny_experiment,
                    paths=paths,
                    representation=representation,
                    history=history,
                    features='both',
                )
                assert {sample.split for sample in data.samples} == {'train', 'val'}
                assert [s.sample_id for s in data.samples] == [
                    s.sample_id for s in blocks['both'].samples
                ]
                assert np.isfinite(data.x).all() and data.x.dtype == np.float32
        with pytest.raises(expected_exception=ValueError):
            load_representation(
                cfg=tiny_experiment,
                paths=paths,
                representation='pixels',
                history='last',
                features='voltage',
            )

    @pytest.mark.parametrize('label_mode', ['true', 'permuted', 'memorization'])
    def test_probe(self, tiny_experiment: ExperimentConfig, label_mode: str) -> None:
        """Run a synthetic probe, inspect split labels, and verify complete persisted results.

        :param tiny_experiment: Offline source experiment.
        :type tiny_experiment: ExperimentConfig
        :param label_mode: Training-label control.
        :type label_mode: str
        """
        paths = get_paths()
        output = paths.runs / label_mode
        result = run_probe(
            cfg=tiny_experiment,
            paths=paths,
            output=output,
            representation='encoded',
            history='all',
            features='both',
            components=10,
            label_mode=label_mode,
        )
        rows = read_table(path=output / 'samples.parquet')
        train = [row for row in rows if row['split'] == 'train']
        assert all(row['split'] != 'test' for row in rows)
        assert all(row['fit_label'] is None for row in rows if row['split'] == 'val')
        assert sorted(row['fit_label'] for row in train) == sorted(row['label'] for row in train)
        if label_mode == 'permuted':
            assert any(row['fit_label'] != row['label'] for row in train)
        if label_mode == 'memorization':
            assert set(result['scores']) == {'memorization_train'}
            assert len(train) == 16
        else:
            assert result['scores']['validation']['n'] == 8
        assert json.loads(s=(output / 'manifest.json').read_text())['status'] == 'completed'
        assert (output / 'checksums.sha256').exists()

    def test_cli(self, tiny_experiment: ExperimentConfig, tmp_path: Path) -> None:
        """Invoke a valid JSON probe, reject invalid configuration, and expose command help.

        :param tiny_experiment: Offline source configuration.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Configuration directory.
        :type tmp_path: Path
        """
        config = tmp_path / 'config.yaml'
        config.write_text(data=effective_yaml(cfg=tiny_experiment))
        runner = CliRunner()
        args = ['diagnose', 'run', str(config), '--output', 'runs/cli', '--json']
        result = runner.invoke(app=app, args=[*args, '--representation', 'pixels'])
        assert result.exit_code == 0, result.output
        assert json.loads(s=result.stdout)['scores']['validation']['n'] == 8
        invalid = runner.invoke(app=app, args=[*args, '--phase', 'invalid'])
        assert invalid.exit_code == 2 and 'error' in json.loads(s=invalid.stdout)
        repeated = runner.invoke(app=app, args=args)
        assert repeated.exit_code == 1 and 'error' in json.loads(s=repeated.stdout)
        assert runner.invoke(app=app, args=['diagnose', 'run', '--help']).exit_code == 0


class TestCohortAudit:
    """Check duplicate classifications and reserve evidence without prediction access."""

    def test_duplicate_classification(self) -> None:
        """Distinguish byte/pixel equality, perceptual candidates, and unrelated signatures."""
        left = {'file_sha256': 'a', 'rgb_sha256': 'b', 'dhash': 0}
        assert duplicate_kind(left=left, right={**left, 'dhash': 255}) == 'exact'
        right = {'file_sha256': 'c', 'rgb_sha256': 'd', 'dhash': 7}
        assert duplicate_kind(left=left, right=right) == 'perceptual_candidate'
        assert duplicate_kind(left=left, right={**right, 'dhash': 255}) is None

    def test_audit(self, tiny_experiment: ExperimentConfig) -> None:
        """Audit a synthetic cohort, then verify graph counts and independent reserve membership.

        :param tiny_experiment: Offline source configuration.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        output = paths.runs / 'audit'
        result = audit_cohort(cfg=tiny_experiment, paths=paths, output=output)
        assert result['cohort_counts'] == {'train': 28, 'val': 8, 'test': 4}
        assert all(row['reserved'] <= 3 for row in result['reserve_coverage'])
        signatures = read_table(path=output / 'source-signatures.parquet')
        historical = {row['filename'] for row in signatures if row['historical']}
        reserve = json.loads(s=(output / 'reserve.json').read_text())
        assert result['reserve_count'] == len(reserve)
        assert not historical.intersection(row['filename'] for row in reserve)
        assert result['graph']['effective_stored_edges'] <= result['graph']['source_stored_edges']
        assert (output / 'source-signatures.parquet').exists()


class TestConvergenceDiagnosis:
    """Preserve failed fold coefficients and keep numerical diagnosis independent of validation."""

    def test_first_failure(self, tiny_experiment: ExperimentConfig) -> None:
        """Exhaust a tiny budget, increase it on the same fold, then verify preserved evidence.

        :param tiny_experiment: Offline cohort and encoder.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        output = paths.runs / 'convergence'
        report = diagnose_convergence(
            cfg=tiny_experiment,
            paths=paths,
            output=output,
            representation='encoded',
            history='last',
            features='both',
            components=10,
            budgets=(1, 1000),
        )
        assert report['failure_reproduced']
        assert report['first_failure']['fold'] == 1
        assert report['first_failure']['training_rows'] == 14
        assert report['validation_predictions_scored'] == 0
        assert not report['budget_measurements'][0]['converged']
        assert report['budget_measurements'][1]['converged']
        assert not list(output.glob('*predictions*'))
        meta = json.loads(s=(output / 'fold-1-budget-1/model.json').read_text())
        assert not meta['converged'] and 'not a selected readout' in meta['purpose']
        assert (output / 'fold-1-budget-1000/weights.npz').is_file()

    @pytest.mark.parametrize('budgets', [(), (0, 1), (2, 1), (1, 1)])
    def test_invalid_budgets(
        self, tiny_experiment: ExperimentConfig, budgets: tuple[int, ...]
    ) -> None:
        """Reject invalid iteration schedules while retaining an immutable failure manifest.

        :param tiny_experiment: Offline source configuration.
        :type tiny_experiment: ExperimentConfig
        :param budgets: Invalid iteration schedule.
        :type budgets: tuple[int, ...]
        """
        paths = get_paths()
        output = paths.runs / 'invalid-convergence'
        with pytest.raises(expected_exception=ValueError):
            diagnose_convergence(
                cfg=tiny_experiment,
                paths=paths,
                output=output,
                representation='pixels',
                history='last',
                features='both',
                components=10,
                budgets=budgets,
            )
        assert json.loads(s=(output / 'manifest.json').read_text())['status'] == 'failed'

    def test_cli(self, tiny_experiment: ExperimentConfig, tmp_path: Path) -> None:
        """Run a convergent small problem through the CLI and reject altered-label diagnosis.

        :param tiny_experiment: Offline cohort and encoder.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Configuration directory.
        :type tmp_path: Path
        """
        config = tmp_path / 'convergence.yaml'
        config.write_text(data=effective_yaml(cfg=tiny_experiment))
        runner = CliRunner()
        args = [
            'diagnose',
            'run',
            str(config),
            '--phase',
            'convergence',
            '--representation',
            'encoded',
            '--components',
            '10',
            '--output',
            'runs/cli-convergence',
            '--json',
        ]
        result = runner.invoke(app=app, args=args)
        assert result.exit_code == 0, result.output
        assert not json.loads(s=result.stdout)['failure_reproduced']
        invalid = runner.invoke(app=app, args=[*args, '--label-mode', 'permuted'])
        assert invalid.exit_code == 2

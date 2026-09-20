"""Compare paired synthetic experiments and exercise scientific compatibility diagnostics."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.evaluation.compare import IncompatibleRunsError, compare, load_evaluation
from flystate.evaluation.evaluate import evaluate
from flystate.experiments.config import ConfigError, ExperimentConfig, effective_yaml
from flystate.readouts.training import train
from flystate.settings import get_paths
from flystate.storage.runs import load_manifest
from flystate.traces.builder import build_trace


@pytest.fixture
def evaluated_runs(tiny_experiment: ExperimentConfig) -> tuple[Path, Path, Path]:
    """Train and evaluate persistent, reset, and altered-PCA reset models on shared samples.

    :param tiny_experiment: Installed offline dataset and connectome.
    :type tiny_experiment: ExperimentConfig
    :returns: Completed evaluated persistent, reset, and altered reset directories.
    :rtype: tuple[Path, Path, Path]
    """
    paths = get_paths()
    runs = []
    for mode, components in (('persistent', 10), ('reset', 10), ('reset', 5)):
        cfg = tiny_experiment.model_copy(
            update={
                'name': f'{mode}-{components}',
                'memory': tiny_experiment.memory.model_copy(update={'mode': mode}),
                'readout': tiny_experiment.readout.model_copy(
                    update={'pca_components': components}
                ),
            }
        )
        build_trace(cfg=cfg, paths=paths)
        result = train(cfg=cfg, paths=paths, original_yaml=effective_yaml(cfg=cfg))
        directory = Path(result['run_dir'])
        evaluate(run_dir=directory, paths=paths)
        runs.append(directory)
    return runs[0], runs[1], runs[2]


class TestComparison:
    """Exercise actual paired runs, report publication, and exact configuration allowances."""

    def test_end_to_end(self, evaluated_runs: tuple[Path, Path, Path]) -> None:
        """Compare compatible runs, inspect reports, then explicitly allow one readout difference.

        :param evaluated_runs: Persistent, reset, and altered-PCA runs.
        :type evaluated_runs: tuple[Path, Path, Path]
        """
        persistent, reset, altered = evaluated_runs
        paths = get_paths()
        runner = CliRunner()
        result = runner.invoke(
            app=app,
            args=[
                'compare',
                persistent.name,
                str(reset),
                '--markdown',
                'reports/paired.md',
                '--json',
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(s=result.stdout)
        assert payload['delta_mem_pp']['diff_pp'] == payload['rows'][-1]['diff_pp']
        assert len(payload['rows']) == 4 and payload['bootstrap'] == 100
        assert set(payload['allowed_differences']) == {'memory.mode', 'name'}
        report = (paths.home / 'reports/paired.md').read_text()
        assert 'Final memory effect' in report and '95% CI (pp)' in report
        artifact = next((paths.runs / 'comparisons').glob('*.json'))
        assert json.loads(s=artifact.read_text()) == payload
        validation = evaluate(run_dir=persistent, paths=paths, split='val')
        selected, _ = load_evaluation(run_dir=persistent)
        assert selected['split'] == 'test'
        with patch(target='flystate.cli.compare.compare', return_value=payload):
            human = runner.invoke(app=app, args=['compare', str(persistent), str(reset)])
        assert human.exit_code == 0 and 'Method' in human.stdout
        mismatched = runner.invoke(
            app=app, args=['compare', str(persistent), str(altered), '--json']
        )
        assert mismatched.exit_code == 3 and 'readout.pca_components' in mismatched.stdout
        allowed = runner.invoke(
            app=app,
            args=[
                'compare',
                str(persistent),
                str(altered),
                '--allow',
                'memory.mode,name,readout.pca_components',
                '--bootstrap',
                '25',
                '--seed',
                '2',
                '--json',
            ],
        )
        assert allowed.exit_code == 0, allowed.output
        assert json.loads(s=allowed.stdout)['bootstrap'] == 25
        same = compare(run_a=reset, run_b=reset, paths=paths)
        assert same['delta_mem_pp'] is None and all(row['diff_pp'] == 0 for row in same['rows'])
        explicit = runner.invoke(
            app=app,
            args=[
                'compare',
                str(persistent),
                str(persistent),
                '--eval-a',
                validation['eval_id'],
                '--eval-b',
                validation['eval_id'],
                '--json',
            ],
        )
        assert explicit.exit_code == 0 and json.loads(s=explicit.stdout)['split'] == 'val'
        assert runner.invoke(app=app, args=['compare', '--help']).exit_code == 0

    def test_diagnostics_and_cli_failures(self, evaluated_runs: tuple[Path, Path, Path]) -> None:
        """Reject all provenance differences together and verify selection and report failure codes.

        :param evaluated_runs: Completed evaluated run directories.
        :type evaluated_runs: tuple[Path, Path, Path]
        """
        persistent, reset, _ = evaluated_runs
        paths = get_paths()
        manifest_path = reset / 'manifest.json'
        manifest = load_manifest(run_dir=reset)
        corrupted = {
            **manifest,
            'threads': 99,
            'dataset_fingerprint': 'wrong',
            'trajectory_hash': 'wrong',
            'brain_files_sha256': {},
        }
        manifest_path.write_text(data=json.dumps(obj=corrupted))
        with pytest.raises(expected_exception=IncompatibleRunsError) as failure:
            compare(run_a=persistent, run_b=reset, paths=paths)
        assert all(
            field in str(failure.value)
            for field in ('threads', 'dataset_fingerprint', 'trajectory_hash', 'brain_files_sha256')
        )
        manifest_path.write_text(data=json.dumps(obj={**manifest, 'status': 'running'}))
        with pytest.raises(expected_exception=ValueError, match='completed runs'):
            compare(run_a=persistent, run_b=reset, paths=paths)
        manifest_path.write_text(data=json.dumps(obj=manifest))
        for allowed, bootstrap, seed in (
            (('unknown',), 100, 0),
            (('name', 'memory.mode'), 0, 0),
            (('name', 'memory.mode'), 100, -1),
        ):
            with pytest.raises(expected_exception=ConfigError):
                compare(
                    run_a=persistent,
                    run_b=reset,
                    paths=paths,
                    allowed=allowed,
                    bootstrap=bootstrap,
                    seed=seed,
                )
        for evaluation_id in ('../outside', 'missing'):
            with pytest.raises(expected_exception=(OSError, ValueError)):
                load_evaluation(run_dir=persistent, evaluation_id=evaluation_id)
        eval_meta, _ = load_evaluation(run_dir=reset)
        eval_path = reset / 'evals' / eval_meta['eval_id'] / 'eval.json'
        eval_path.write_text(data=json.dumps(obj={**eval_meta, 'status': 'failed'}))
        with pytest.raises(expected_exception=ValueError, match='incomplete'):
            load_evaluation(run_dir=reset, evaluation_id=eval_meta['eval_id'])
        with pytest.raises(expected_exception=ValueError, match='no completed test'):
            load_evaluation(run_dir=reset)
        eval_path.write_text(data=json.dumps(obj=eval_meta))
        meta_a, records_a = load_evaluation(run_dir=persistent)
        meta_b, records_b = load_evaluation(run_dir=reset)
        changed_labels = [
            {**row, 'y_true': row['y_true'] + 100, 'correct': False} for row in records_b
        ]
        different_samples = [
            {**row, 'sample_id': row['sample_id'] + '-different'} for row in records_b
        ]
        for altered_meta, altered_records, message in (
            ({**meta_b, 'split': 'val'}, records_b, 'evaluation.split'),
            (meta_b, records_b[:-1], 'incomplete'),
            (meta_b, different_samples, 'identities differ'),
            (meta_b, changed_labels, 'true labels differ'),
        ):
            with (
                patch(
                    target='flystate.evaluation.compare.load_evaluation',
                    side_effect=[(meta_a, records_a), (altered_meta, altered_records)],
                ),
                pytest.raises(expected_exception=IncompatibleRunsError, match=message),
            ):
                compare(run_a=persistent, run_b=reset, paths=paths)
        runner = CliRunner()
        for args in (['--bootstrap', '0'], ['--seed', '-1']):
            invalid = runner.invoke(
                app=app, args=['compare', str(persistent), str(reset), *args, '--json']
            )
            assert invalid.exit_code == 2
        existing = paths.home / 'existing.md'
        existing.write_text(data='preserve')
        error = runner.invoke(
            app=app, args=['compare', str(persistent), str(reset), '--markdown', str(existing)]
        )
        assert error.exit_code == 1 and error.stdout == '' and existing.read_text() == 'preserve'
        with patch(target='flystate.cli.compare.compare', side_effect=KeyboardInterrupt):
            interrupted = runner.invoke(
                app=app, args=['compare', str(persistent), str(reset), '--json']
            )
        assert interrupted.exit_code == 130

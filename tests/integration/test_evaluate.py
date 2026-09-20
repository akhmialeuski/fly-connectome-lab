"""Verify reproducible append-only evaluation and reject inconsistent recorded artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.evaluation.evaluate import evaluate, evaluation_inputs
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.readouts.training import train
from flystate.settings import get_paths
from flystate.storage.parquet import read_table, write_table
from flystate.storage.runs import RunError, load_manifest
from flystate.traces.builder import build_trace


@pytest.fixture
def trained_run(tiny_experiment: ExperimentConfig, request: pytest.FixtureRequest) -> Path:
    """Build and train one complete offline experiment for evaluation checks.

    :param tiny_experiment: Installed synthetic dataset and connectome.
    :type tiny_experiment: ExperimentConfig
    :param request: Optional indirect memory-policy parameter.
    :type request: pytest.FixtureRequest
    :returns: Completed immutable training directory.
    :rtype: Path
    """
    paths = get_paths()
    cfg = tiny_experiment.model_copy(
        update={
            'memory': tiny_experiment.memory.model_copy(
                update={'mode': getattr(request, 'param', 'persistent')}
            )
        }
    )
    build_trace(cfg=cfg, paths=paths)
    result = train(cfg=cfg, paths=paths, original_yaml=effective_yaml(cfg=cfg))
    return Path(result['run_dir'])


class TestEvaluation:
    """Validate typed predictions, exact validation reproduction, and run immutability."""

    @pytest.mark.parametrize('trained_run', ['persistent', 'reset_concat'], indirect=True)
    def test_artifacts_and_reproduction(self, trained_run: Path) -> None:
        """Evaluate test and validation, inspect predictions, then verify original file timestamps.

        :param trained_run: Completed synthetic training run.
        :type trained_run: Path
        """
        original = {
            path: (path.stat().st_mtime_ns, path.read_bytes())
            for path in trained_run.rglob('*')
            if path.is_file()
        }
        runner = CliRunner()
        with patch(target='flystate.evaluation.evaluate.datetime') as clock:
            clock.now.return_value = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
            first = runner.invoke(app=app, args=['evaluate', str(trained_run), '--json'])
            second = runner.invoke(app=app, args=['evaluate', trained_run.name, '--json'])
        assert first.exit_code == 0, first.output
        assert second.exit_code == 0, second.output
        result = json.loads(s=first.stdout)
        assert result['eval_id'] != json.loads(s=second.stdout)['eval_id']
        directory = Path(result['eval_dir'])
        assert {path.name for path in directory.iterdir()} == {
            'eval.json',
            'predictions.parquet',
            'metrics.parquet',
            'confusion.parquet',
            'timings.parquet',
        }
        metadata = json.loads(s=(directory / 'eval.json').read_text())
        assert metadata['status'] == 'completed' and metadata['n_samples'] == 4
        table = pq.read_table(source=directory / 'predictions.parquet')
        assert table.schema.field('t').type == pa.int16()
        assert table.schema.field('y_true').type == pa.int32()
        assert table.schema.field('p_pred').type == pa.float32()
        assert table.schema.field('top5_labels').type == pa.list_(value_type=pa.int32())
        assert table.schema.field('top5_probs').type == pa.list_(value_type=pa.float32())
        predictions = table.to_pylist()
        assert len(predictions) == 16
        assert len({(row['sample_id'], row['t']) for row in predictions}) == 16
        metrics = read_table(path=directory / 'metrics.parquet')
        for row in metrics:
            observed = [prediction for prediction in predictions if prediction['t'] == row['t']]
            assert row['accuracy'] == np.mean(a=[prediction['correct'] for prediction in observed])
            assert row['top5_accuracy'] >= row['accuracy']
            assert row['ci_low'] <= row['accuracy'] <= row['ci_high']
        for prediction in predictions:
            assert prediction['p_pred'] == prediction['top5_probs'][0]
            assert prediction['y_pred'] == prediction['top5_labels'][0]
            assert 0 <= prediction['p_true'] <= 1
        confusion = read_table(path=directory / 'confusion.parquet')
        assert sum(row['count'] for row in confusion) == 4
        assert (
            sum(row['count'] for row in confusion if row['y_true'] == row['y_pred'])
            == 4 * metrics[-1]['accuracy']
        )
        timings = read_table(path=directory / 'timings.parquet')
        assert len(timings) == 4 and all(
            row['readout_ms'] > 0 and row['brain_ms'] >= 0 for row in timings
        )
        validation = runner.invoke(
            app=app, args=['evaluate', str(trained_run), '--split', 'val', '--json']
        )
        assert validation.exit_code == 0, validation.output
        actual = json.loads(s=validation.stdout)
        training = read_table(path=trained_run / 'metrics/validation.parquet')
        assert actual['accuracy_by_t'] == {str(row['t']): row['accuracy'] for row in training}
        assert all(
            (path.stat().st_mtime_ns, path.read_bytes()) == state
            for path, state in original.items()
        )
        with patch(target='flystate.cli.evaluate.evaluate', return_value=result):
            human = runner.invoke(app=app, args=['evaluate', str(trained_run)])
        assert human.exit_code == 0 and 'Wilson interval' in human.stdout
        assert runner.invoke(app=app, args=['evaluate', '--help']).exit_code == 0

    def test_provenance_rejection(self, trained_run: Path) -> None:
        """Tamper with lifecycle and provenance fields, reject them, then restore each artifact.

        :param trained_run: Completed synthetic training run to corrupt temporarily.
        :type trained_run: Path
        """
        manifest_path = trained_run / 'manifest.json'
        manifest = load_manifest(run_dir=trained_run)
        for field, value in (
            ('status', 'running'),
            ('config_hash', 'wrong'),
            ('cache_key', '../outside'),
            ('threads', 2),
            ('trajectory_hash', 'wrong'),
        ):
            manifest_path.write_text(data=json.dumps(obj={**manifest, field: value}))
            with pytest.raises(expected_exception=RunError):
                evaluation_inputs(run_dir=trained_run, paths=get_paths())
        manifest_path.write_text(data=json.dumps(obj=manifest))
        model_path = trained_run / 'model/model.json'
        model = json.loads(s=model_path.read_text())
        model_path.write_text(data=json.dumps(obj={**model, 'mode': 'reset'}))
        with pytest.raises(expected_exception=RunError, match='Readout geometry'):
            evaluation_inputs(run_dir=trained_run, paths=get_paths())
        model_path.write_text(data=json.dumps(obj=model))
        cfg, store, models = evaluation_inputs(run_dir=trained_run, paths=get_paths())
        assert store.meta['key'] == manifest['cache_key']
        store.index = [row for row in store.index if row['split'] != 'test']
        with (
            patch(
                target='flystate.evaluation.evaluate.evaluation_inputs',
                return_value=(cfg, store, models),
            ),
            pytest.raises(expected_exception=ValueError, match='No samples'),
        ):
            evaluate(run_dir=trained_run, paths=get_paths())
        with pytest.raises(expected_exception=ValueError, match='split'):
            evaluate(run_dir=trained_run, paths=get_paths(), split='train')
        assert not (trained_run / 'evals').exists()

    def test_failure_status_and_cli(self, trained_run: Path) -> None:
        """Preserve failed evaluation status and expose runtime, argument, and interrupt exit codes.

        :param trained_run: Completed synthetic training run.
        :type trained_run: Path
        """
        with (
            patch(
                target='flystate.evaluation.evaluate.write_table', side_effect=OSError('disk full')
            ),
            pytest.raises(expected_exception=OSError, match='disk full'),
        ):
            evaluate(run_dir=trained_run, paths=get_paths())
        metadata_path = next((trained_run / 'evals').glob('*/eval.json'))
        assert json.loads(s=metadata_path.read_text())['status'] == 'failed'
        assert load_manifest(run_dir=trained_run)['status'] == 'completed'
        runner = CliRunner()
        invalid = runner.invoke(
            app=app, args=['evaluate', str(trained_run), '--split', 'train', '--json']
        )
        assert invalid.exit_code == 2 and 'error' in json.loads(s=invalid.stdout)
        manifest_path = trained_run / 'manifest.json'
        manifest = load_manifest(run_dir=trained_run)
        manifest_path.write_text(data=json.dumps(obj={**manifest, 'status': 'running'}))
        running = runner.invoke(app=app, args=['evaluate', str(trained_run), '--json'])
        assert running.exit_code == 1 and 'completed' in running.stdout
        manifest_path.write_text(data=json.dumps(obj=manifest))
        with patch(target='flystate.cli.evaluate.evaluate', side_effect=KeyboardInterrupt):
            interrupted = runner.invoke(app=app, args=['evaluate', str(trained_run), '--json'])
        assert interrupted.exit_code == 130
        failed = runner.invoke(app=app, args=['evaluate', 'missing'])
        assert failed.exit_code == 1 and failed.stdout == ''


class TestParquetStorage:
    """Check typed atomic publication and cleanup without touching existing artifacts."""

    def test_write_failures(self, tmp_path: Path) -> None:
        """Publish a table, reject replacement, then ensure failed writes leave no temporary files.

        :param tmp_path: Isolated output directory.
        :type tmp_path: Path
        """
        path = tmp_path / 'table.parquet'
        write_table(path=path, rows=[{'value': 1}])
        assert read_table(path=path) == [{'value': 1}]
        with pytest.raises(expected_exception=FileExistsError):
            write_table(path=path, rows=[{'value': 2}])
        with (
            patch(target='flystate.storage.parquet.pq.write_table', side_effect=OSError('disk')),
            pytest.raises(expected_exception=OSError),
        ):
            write_table(path=tmp_path / 'failed.parquet', rows=[{'value': 1}])
        assert list(tmp_path.iterdir()) == [path]

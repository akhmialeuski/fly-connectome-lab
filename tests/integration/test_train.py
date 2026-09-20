"""Exercise offline trace-to-readout runs and verify reproducible validation artifacts."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.readouts.linear import features_at, load_readouts
from flystate.readouts.training import accuracy_metrics, read_feature_rows, train
from flystate.settings import get_paths
from flystate.storage.runs import load_manifest, update_manifest
from flystate.traces.builder import build_trace, open_trace


class TestTraining:
    """Check all memory policies, run contracts, and failures using real synthetic simulation."""

    @pytest.mark.parametrize('mode', ['persistent', 'reset', 'reset_concat'])
    def test_end_to_end(self, tiny_experiment: ExperimentConfig, tmp_path: Path, mode: str) -> None:
        """Build a trace, train via CLI, load its models, and reproduce each validation metric.

        :param tiny_experiment: Installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Test configuration directory.
        :type tmp_path: Path
        :param mode: Readout memory policy.
        :type mode: str
        """
        cfg = tiny_experiment.model_copy(
            update={'memory': tiny_experiment.memory.model_copy(update={'mode': mode})}
        )
        paths = get_paths()
        build_trace(cfg=cfg, paths=paths)
        config = tmp_path / 'train.yaml'
        original = effective_yaml(cfg=cfg)
        config.write_text(data=original)
        runner = CliRunner()
        with patch(
            target='flystate.readouts.training.read_feature_rows', wraps=read_feature_rows
        ) as reader:
            result = runner.invoke(
                app=app, args=['train', str(config), '--set', 'name=trained', '--json']
            )
        assert result.exit_code == 0, result.output
        output = json.loads(s=result.stdout)
        run_dir = Path(output['run_dir'])
        assert reader.call_count == 2
        assert all(call.kwargs['rows'].max() < 36 for call in reader.call_args_list)
        manifest = load_manifest(run_dir=run_dir)
        assert manifest['status'] == 'completed' and manifest['threads'] == 1
        assert manifest['finished_utc'] is not None and manifest['parent_run_id'] is None
        expected = [
            'config.original.yaml',
            'config.yaml',
            'environment.json',
            'manifest.json',
            'summary.json',
            'logs/events.jsonl',
            'model/weights.npz',
            'model/model.json',
            'metrics/validation.parquet',
        ]
        assert all((run_dir / name).is_file() for name in expected)
        assert (run_dir / 'config.original.yaml').read_text() == original
        assert 'name: trained' in (run_dir / 'config.yaml').read_text()
        models = load_readouts(path=run_dir / 'model')
        store = open_trace(cfg=cfg, paths=paths)
        rows = np.asarray(a=[row['row'] for row in store.index if row['split'] == 'val'])
        features = read_feature_rows(store=store, rows=rows)
        targets = np.asarray(a=[store.index[int(row)]['label'] for row in rows])
        metrics = pq.read_table(source=run_dir / 'metrics/validation.parquet').to_pylist()
        assert len(metrics) == 4 and len(models) == 4
        for step, model in enumerate(models, start=1):
            probabilities = model.predict_proba(x=features_at(features=features, t=step, mode=mode))
            actual = accuracy_metrics(
                probabilities=probabilities, targets=targets, classes=model.classes
            )
            assert actual['accuracy'] == metrics[step - 1]['accuracy']
            assert actual['top5_accuracy'] == 1
        with patch(target='flystate.cli.train.train', return_value=output):
            human = runner.invoke(app=app, args=['train', str(config)])
        assert human.exit_code == 0 and 'validation accuracy' in human.stdout

    def test_failure_and_cli_errors(
        self, tiny_experiment: ExperimentConfig, tmp_path: Path
    ) -> None:
        """Reject missing caches, preserve failed runs, and expose configuration/runtime exit codes.

        :param tiny_experiment: Installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Test configuration directory.
        :type tmp_path: Path
        """
        paths = get_paths()
        config = tmp_path / 'train.yaml'
        config.write_text(data=effective_yaml(cfg=tiny_experiment))
        runner = CliRunner()
        missing = runner.invoke(app=app, args=['train', str(config), '--json'])
        assert missing.exit_code == 1 and 'flystate trace build' in missing.stdout
        invalid = runner.invoke(app=app, args=['train', str(tmp_path / 'absent'), '--json'])
        assert invalid.exit_code == 2
        build_trace(cfg=tiny_experiment, paths=paths)
        with (
            patch(
                target='flystate.readouts.training.fit_readout',
                side_effect=RuntimeError('fit failed'),
            ),
            pytest.raises(expected_exception=RuntimeError, match='fit failed'),
        ):
            train(cfg=tiny_experiment, paths=paths, original_yaml=config.read_text())
        manifests = list(paths.runs.glob('*/manifest.json'))
        assert len(manifests) == 1
        assert load_manifest(run_dir=manifests[0].parent)['status'] == 'failed'
        with patch(target='flystate.cli.train.train', side_effect=KeyboardInterrupt):
            interrupted = runner.invoke(app=app, args=['train', str(config), '--json'])
        assert interrupted.exit_code == 130
        with patch(target='flystate.cli.train.train', side_effect=ValueError('invalid')):
            error = runner.invoke(app=app, args=['train', str(config)])
        assert error.exit_code == 1 and error.stdout == ''

    def test_interrupt_after_completion(self, tiny_experiment: ExperimentConfig) -> None:
        """Preserve completed artifacts if interrupted immediately after publishing completion.

        :param tiny_experiment: Installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        build_trace(cfg=tiny_experiment, paths=paths)

        def complete_then_interrupt(run_dir: Path, **fields: object) -> None:
            """Publish the actual lifecycle transition and simulate a subsequent interrupt.

            :param run_dir: Training run being completed.
            :type run_dir: Path
            :param fields: Lifecycle updates to publish.
            :type fields: object
            :raises KeyboardInterrupt: After the manifest is durably completed.
            """
            update_manifest(run_dir=run_dir, **fields)
            raise KeyboardInterrupt

        with (
            patch(
                target='flystate.readouts.training.update_manifest',
                side_effect=complete_then_interrupt,
            ),
            pytest.raises(expected_exception=KeyboardInterrupt),
        ):
            train(cfg=tiny_experiment, paths=paths, original_yaml='')
        manifest = next(paths.runs.glob('*/manifest.json'))
        assert load_manifest(run_dir=manifest.parent)['status'] == 'completed'

    @pytest.mark.parametrize(
        'probabilities,targets,classes',
        [
            (np.asarray([[0.5, 0.5]]), np.asarray([[0]]), np.asarray([0, 1])),
            (np.asarray([[0.5, 0.5]]), np.asarray([0]), np.asarray([[0, 1]])),
            (np.asarray([[0.5, 0.5]]), np.asarray([0]), np.asarray([0, 0])),
            (np.asarray([[1.2, -0.2]]), np.asarray([0]), np.asarray([0, 1])),
            (np.asarray([[0.2, 0.2]]), np.asarray([0]), np.asarray([0, 1])),
        ],
    )
    def test_invalid_metrics(
        self, probabilities: np.ndarray, targets: np.ndarray, classes: np.ndarray
    ) -> None:
        """Reject broadcasting hazards, duplicate classes, and invalid probability distributions.

        :param probabilities: Invalid distribution or valid data paired with malformed labels.
        :type probabilities: np.ndarray
        :param targets: Target labels.
        :type targets: np.ndarray
        :param classes: Probability column labels.
        :type classes: np.ndarray
        """
        with pytest.raises(expected_exception=ValueError):
            accuracy_metrics(probabilities=probabilities, targets=targets, classes=classes)

    def test_row_selection_and_metrics(self, tiny_experiment: ExperimentConfig) -> None:
        """Reject invalid row selectors and score top-five predictions deterministically.

        :param tiny_experiment: Installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        build_trace(cfg=tiny_experiment, paths=get_paths())
        store = open_trace(cfg=tiny_experiment, paths=get_paths())
        for rows in (
            np.asarray([2, 1]),
            np.asarray([-1]),
            np.asarray([40]),
            np.asarray([1.0, 2.0]),
            np.asarray([[1]]),
        ):
            with pytest.raises(expected_exception=ValueError):
                read_feature_rows(store=store, rows=rows)
        probabilities = np.asarray(
            a=[[0.6, 0.1, 0.1, 0.1, 0.05, 0.05], [0.05, 0.05, 0.1, 0.1, 0.1, 0.6]]
        )
        result = accuracy_metrics(
            probabilities=probabilities, targets=np.asarray([0, 0]), classes=np.arange(6)
        )
        assert result['accuracy'] == 0.5 and result['top5_accuracy'] == 1
        with pytest.raises(expected_exception=ValueError):
            accuracy_metrics(
                probabilities=probabilities, targets=np.asarray([8, 8]), classes=np.arange(6)
            )

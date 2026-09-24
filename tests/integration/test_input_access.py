"""Offline T29 fit/query isolation, evidence persistence, and CLI contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets.preprocess import PreparedDataset
from flystate.diagnostics import input_access
from flystate.diagnostics.probes import score_split as real_score_split
from flystate.experiments.config import ExperimentConfig, effective_yaml, load_config
from flystate.settings import Paths, get_paths
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table


def test_frozen_json_grid_matches_effective_tuple() -> None:
    """Accept the real frozen C grid but reject a changed numeric candidate."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(path=root / 'configs/celeba-smoke.yaml')
    study = root / 'research/sequential-visual-memory/2026-09-24-input-access'
    cohort = json.loads(s=(study / 'cohort.json').read_text(encoding='utf-8'))
    schedule = json.loads(s=(study / 'schedule.json').read_text(encoding='utf-8'))
    assert isinstance(cfg.readout.c_grid, tuple)
    assert isinstance(schedule['readout']['c_grid'], list)
    input_access._verify_readout_schedule(cfg=cfg, cohort=cohort, schedule=schedule)
    schedule['readout']['c_grid'][0] = 0.02
    with pytest.raises(expected_exception=ValueError, match='readout or gate'):
        input_access._verify_readout_schedule(cfg=cfg, cohort=cohort, schedule=schedule)


def test_query_scoring_follows_both_fits(
    tiny_experiment: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep query scores behind both fits and persist immutable numeric evidence.

    :param tiny_experiment: Offline fixture configuration and data home.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Isolated protocol files.
    :type tmp_path: Path
    :param monkeypatch: Scoped source and score spies.
    :type monkeypatch: pytest.MonkeyPatch
    """
    items = [
        {
            'sample_id': f'sample-{label:02d}-{rank:02d}',
            'label': label,
            'role': 'fit' if rank < 10 else 'query',
        }
        for label in range(20)
        for rank in range(14)
    ]
    cohort = tmp_path / 'cohort.json'
    schedule = tmp_path / 'schedule.json'
    membership = tmp_path / 'membership.json'
    write_json(path=cohort, value={'samples': items, 'source_membership_sha256': 'source'})
    write_json(
        path=schedule,
        value={
            'readout': {
                'pca_components': 10,
                'c_grid': [0.1],
                'cv_folds': 2,
                'seed': 0,
                'logistic_tolerance': 1e-4,
                'max_iterations': 5000,
            },
            'resource_limit': {'max_wall_minutes': 60, 'max_rss_gib': 12},
        },
    )
    write_json(path=membership, value={})
    prepared = PreparedDataset(
        key='synthetic',
        directory=tmp_path,
        samples=[],
        images=np.empty(shape=(0, 1, 1, 3), dtype=np.uint8),
        template=np.zeros(shape=(5, 2), dtype=np.float64),
        fingerprint='synthetic',
    )

    def fake_prepare(cfg: ExperimentConfig, paths: Paths) -> PreparedDataset:
        """Return an offline aligned-cache handle.

        :param cfg: Unused fixture configuration.
        :type cfg: ExperimentConfig
        :param paths: Unused isolated home.
        :type paths: Paths
        :returns: Empty prepared-data handle for the mocked feature boundary.
        :rtype: PreparedDataset
        """
        return prepared

    def fake_verify(**kwargs: Any) -> list[int]:
        """Accept the test-owned protocol at the source-verification boundary.

        :param kwargs: Protocol verification inputs.
        :type kwargs: Any
        :returns: 280 synthetic feature indices.
        :rtype: list[int]
        """
        return list(range(280))

    def fake_features(
        cfg: ExperimentConfig, paths: Paths, prepared: PreparedDataset, rows: list[int]
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Give both probes small, deterministic class-coded matrices.

        :param cfg: Unused fixture configuration.
        :type cfg: ExperimentConfig
        :param paths: Unused isolated home.
        :type paths: Paths
        :param prepared: Empty cache boundary handle.
        :type prepared: PreparedDataset
        :param rows: Ordered 280 synthetic feature indices.
        :type rows: list[int]
        :returns: Two independent finite float32 feature matrices.
        :rtype: tuple[dict[str, NDArray[np.float32]], dict[str, Any]]
        """
        assert len(rows) == 280
        matrix = np.zeros(shape=(280, 20), dtype=np.float32)
        matrix[np.arange(280), np.repeat(np.arange(20), 14)] = 1
        return {'pixels': matrix, 'encoded_current': matrix.copy()}, {'synthetic': True}

    monkeypatch.setattr(target=input_access, name='prepare_dataset', value=fake_prepare)
    monkeypatch.setattr(target=input_access, name='_verify_protocol', value=fake_verify)
    monkeypatch.setattr(target=input_access, name='_features', value=fake_features)
    original_fit = input_access.fit_classifier
    completed_fits: list[int] = []

    def counted_fit(**kwargs: Any) -> Any:
        """Count actual fold-local fits before allowing any query score.

        :param kwargs: Forwarded fit parameters.
        :type kwargs: Any
        :returns: Fitted pipeline and CV means.
        :rtype: Any
        """
        result = original_fit(**kwargs)
        completed_fits.append(len(completed_fits) + 1)
        return result

    def guarded_score(**kwargs: Any) -> dict[str, Any]:
        """Assert both fits completed before even in-sample scoring.

        :param kwargs: Forwarded score parameters.
        :type kwargs: Any
        :returns: Persisted split score.
        :rtype: dict[str, Any]
        """
        assert len(completed_fits) == 2
        return real_score_split(**kwargs)

    monkeypatch.setattr(target=input_access, name='fit_classifier', value=counted_fit)
    monkeypatch.setattr(target=input_access, name='score_split', value=guarded_score)
    output = get_paths().runs / 'diagnostics' / 'input-access-offline'
    report = input_access.run_input_access(
        cfg=tiny_experiment,
        paths=get_paths(),
        output=output,
        cohort_path=cohort,
        schedule_path=schedule,
        membership_path=membership,
    )
    assert completed_fits == [1, 2]
    assert report['cohort_counts'] == {'fit': 200, 'query': 80, 'classes': 20}
    assert len(read_table(path=output / 'metrics.parquet')) == 4
    for name in input_access.REPRESENTATIONS:
        assert (output / name / 'model' / 'weights.npz').exists()
        assert len(read_table(path=output / name / 'query-predictions.parquet')) == 80
    manifest = json.loads(s=(output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'


def test_input_access_cli_json_and_help(
    tiny_experiment: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the new command with help and exactly one JSON result or error object.

    :param tiny_experiment: Offline valid config.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Isolated configuration file location.
    :type tmp_path: Path
    :param monkeypatch: Scoped implementation replacement.
    :type monkeypatch: pytest.MonkeyPatch
    """
    config = tmp_path / 'config.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    runner = CliRunner()
    help_result = runner.invoke(app=app, args=['diagnose', 'input-access', '--help'])
    assert help_result.exit_code == 0
    assert '--cohort' in help_result.stdout and '--json' in help_result.stdout

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        """Return a small successful command result without external data.

        :param kwargs: Forwarded CLI inputs.
        :type kwargs: Any
        :returns: Synthetic result object.
        :rtype: dict[str, Any]
        """
        assert kwargs['cfg'].name == tiny_experiment.name
        return {'gate_passed': False}

    monkeypatch.setattr('flystate.cli.input_access.run_input_access', fake_run)
    args = [
        'diagnose',
        'input-access',
        str(config),
        '--output',
        'runs/offline',
        '--cohort',
        str(tmp_path / 'cohort.json'),
        '--schedule',
        str(tmp_path / 'schedule.json'),
        '--membership',
        str(tmp_path / 'membership.json'),
        '--json',
    ]
    result = runner.invoke(app=app, args=args)
    assert result.exit_code == 0
    assert json.loads(s=result.stdout) == {'gate_passed': False}

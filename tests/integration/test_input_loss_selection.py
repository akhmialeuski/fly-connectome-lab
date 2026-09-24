"""Offline T30 fold isolation, frozen protocol, and CLI checks."""

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics import input_loss_selection
from flystate.experiments.config import ExperimentConfig, effective_yaml, load_config


def test_frozen_t30_schedule_and_source_hashes() -> None:
    """Accept the committed parent evidence and reject a changed C candidate."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(path=root / 'configs/celeba-smoke.yaml')
    parent = root / 'research/sequential-visual-memory/2026-09-24-input-access'
    study = root / 'research/sequential-visual-memory/2026-09-24-input-loss-selection'
    schedule = json.loads(s=(study / 'schedule.json').read_text(encoding='utf-8'))
    input_loss_selection._verify_schedule(
        cfg=cfg,
        schedule=schedule,
        cohort_path=parent / 'cohort.json',
        parent_schedule_path=parent / 'schedule.json',
        baseline_path=parent / 'snapshot/A1',
    )
    schedule['candidate_C'][0] = 0.02
    with pytest.raises(expected_exception=ValueError, match='frozen T30 schedule'):
        input_loss_selection._verify_schedule(
            cfg=cfg,
            schedule=schedule,
            cohort_path=parent / 'cohort.json',
            parent_schedule_path=parent / 'schedule.json',
            baseline_path=parent / 'snapshot/A1',
        )


def test_all_candidates_have_exact_fit_only_oof_rows() -> None:
    """Save one probability per fit image and C, and reject shifted folds."""
    labels = np.repeat(np.arange(20, dtype=np.int64), 10)
    ids = [f'fit-{index:03d}' for index in range(200)]
    x = np.eye(20, dtype=np.float32)[labels]
    folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for index, (training, held_out) in enumerate(splitter.split(X=x, y=labels)):
        folds.append(
            {
                'index': index,
                'fit_ids': [ids[i] for i in training],
                'held_out_ids': [ids[i] for i in held_out],
                'pca_components': 20,
            }
        )
    schedule = {
        'candidate_C': [0.01, 0.1],
        'pca_components': 20,
        'seed': 0,
        'tolerance': 1e-4,
        'max_iterations': 5000,
        'resources': {'max_wall_minutes': 60, 'max_peak_rss_gib': 12},
    }
    oof, fold_rows, candidates, selected = input_loss_selection._cross_validate(
        x=x,
        labels=labels,
        ids=ids,
        schedule=schedule,
        parent_folds=folds,
        representation='pixels',
        started=perf_counter(),
    )
    assert len(oof) == 400
    assert len(fold_rows) == 10
    assert len(candidates) == 2
    assert selected == min(candidates, key=lambda row: (row['log_loss'], row['C']))['C']
    assert {row['sample_id'] for row in oof} == set(ids)
    assert all(len(row['probabilities']) == 20 for row in oof)
    assert all(row['accuracy'] == 1 for row in candidates)
    folds[0]['held_out_ids'][0] = 'wrong-photo'
    with pytest.raises(expected_exception=ValueError, match='fold membership'):
        input_loss_selection._cross_validate(
            x=x,
            labels=labels,
            ids=ids,
            schedule=schedule,
            parent_folds=folds,
            representation='pixels',
            started=perf_counter(),
        )


def test_input_loss_selection_cli_help_and_json(
    tiny_experiment: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose help and one stdout JSON object through Typer's actual CLI.

    :param tiny_experiment: Offline synthetic experiment configuration.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Isolated config-file location.
    :type tmp_path: Path
    :param monkeypatch: Scoped diagnostic implementation stub.
    :type monkeypatch: pytest.MonkeyPatch
    """
    config = tmp_path / 'config.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    runner = CliRunner()
    help_result = runner.invoke(app=app, args=['diagnose', 'input-loss-selection', '--help'])
    assert help_result.exit_code == 0
    visible_help = re.sub(pattern=r'\x1b\[[0-9;]*m', repl='', string=help_result.stdout)
    assert '--parent-schedule' in visible_help and '--json' in visible_help

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        """Return a minimal valid command object without source files.

        :param kwargs: Forwarded CLI inputs.
        :type kwargs: Any
        :returns: Synthetic scientific result.
        :rtype: dict[str, Any]
        """
        assert kwargs['cfg'].name == tiny_experiment.name
        return {'gate_passed': True}

    monkeypatch.setattr('flystate.cli.input_loss_selection.run_input_loss_selection', fake_run)
    args = [
        'diagnose',
        'input-loss-selection',
        str(config),
        '--output',
        'runs/offline',
        '--cohort',
        str(tmp_path / 'cohort.json'),
        '--parent-schedule',
        str(tmp_path / 'parent-schedule.json'),
        '--membership',
        str(tmp_path / 'membership.json'),
        '--baseline',
        str(tmp_path / 'baseline'),
        '--schedule',
        str(tmp_path / 'schedule.json'),
        '--json',
    ]
    result = runner.invoke(app=app, args=args)
    assert result.exit_code == 0
    assert json.loads(s=result.stdout) == {'gate_passed': True}

"""Offline T31 source freezing, paired cluster inference, and CLI contracts."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics import matched_neural
from flystate.experiments.config import ExperimentConfig, effective_yaml, load_config
from flystate.hashing import stable_int


def test_frozen_source_schedule_rejects_changed_candidate() -> None:
    """Accept the committed parent hashes and reject a changed C value."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(path=root / 'configs/celeba-smoke.yaml')
    t29 = root / 'research/sequential-visual-memory/2026-09-24-input-access'
    t30 = root / 'research/sequential-visual-memory/2026-09-24-input-loss-selection'
    t31 = root / 'research/sequential-visual-memory/2026-09-24-matched-neural-access'
    schedule = json.loads(s=(t31 / 'schedule.json').read_text(encoding='utf-8'))
    arguments = {
        'cfg': cfg,
        'schedule': schedule,
        'cohort_path': t29 / 'cohort.json',
        'parent_schedule_path': t29 / 'schedule.json',
        't30_schedule_path': t30 / 'schedule.json',
        't30_path': t30 / 'snapshot/A0',
        'membership_path': root
        / 'research/sequential-visual-memory/2026-09-23-noise-recognition/membership.json',
    }
    matched_neural._verify_schedule(**arguments)
    schedule['candidate_C'][0] = 0.02
    with pytest.raises(expected_exception=ValueError, match='frozen T31 schedule'):
        matched_neural._verify_schedule(**arguments)


def test_paired_identity_clusters_and_alignment() -> None:
    """Bootstrap whole identities and reject mismatched photo order."""
    input_rows = []
    neural_rows = []
    for label in range(20):
        for rank in range(10):
            sample_id = f'fit-{label:02d}-{rank:02d}'
            input_rows.append(
                {
                    'sample_id': sample_id,
                    'y_true': label,
                    'y_pred': label if label < 10 else (label + 1) % 20,
                }
            )
            neural_rows.append(
                {
                    'sample_id': sample_id,
                    'y_true': label,
                    'y_pred': label if label < 5 else (label + 1) % 20,
                }
            )
    generator = np.random.default_rng(
        seed=np.random.SeedSequence(entropy=[0, stable_int(key='t31-identity-bootstrap')])
    )
    draws = generator.integers(low=0, high=20, size=(10000, 20), dtype=np.int64)
    summary, paired, bootstrap = matched_neural._paired_analysis(
        case='N0',
        input_rows=input_rows,
        neural_rows=neural_rows,
        input_metrics={'accuracy': 0.5, 'log_loss': 2.0},
        neural_metrics={'accuracy': 0.25, 'log_loss': 3.0},
        bootstrap_draws=draws,
    )
    assert summary['neural_minus_input_accuracy_pp'] == -25.0
    assert summary['cluster_interval_high_pp'] < 0
    assert summary['error_overlap'] == {
        'both_correct': 50,
        'input_only': 50,
        'neural_only': 0,
        'both_wrong': 100,
    }
    assert len(paired) == 200 and len(bootstrap) == 10000
    neural_rows[0]['sample_id'] = 'different'
    with pytest.raises(expected_exception=ValueError, match='not paired'):
        matched_neural._paired_analysis(
            case='N0',
            input_rows=input_rows,
            neural_rows=neural_rows,
            input_metrics={'accuracy': 0.5, 'log_loss': 2.0},
            neural_metrics={'accuracy': 0.25, 'log_loss': 3.0},
            bootstrap_draws=draws,
        )


def test_matched_neural_cli_help_and_json(
    tiny_experiment: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke the real command route with help and one JSON stdout object.

    :param tiny_experiment: Offline synthetic experiment configuration.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Isolated config-file location.
    :type tmp_path: Path
    :param monkeypatch: Scoped scientific implementation replacement.
    :type monkeypatch: pytest.MonkeyPatch
    """
    config = tmp_path / 'config.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    runner = CliRunner()
    help_result = runner.invoke(app=app, args=['diagnose', 'matched-neural', '--help'])
    assert help_result.exit_code == 0
    visible_help = re.sub(pattern=r'\x1b\[[0-9;]*m', repl='', string=help_result.stdout)
    assert '--input-attempt' in visible_help and '--json' in visible_help

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        """Return a small result without loading external brain or face data.

        :param kwargs: Forwarded CLI inputs.
        :type kwargs: Any
        :returns: Synthetic matched comparison result.
        :rtype: dict[str, Any]
        """
        assert kwargs['cfg'].name == tiny_experiment.name
        return {'gate_passed': False}

    monkeypatch.setattr('flystate.cli.matched_neural.run_matched_neural', fake_run)
    args = [
        'diagnose',
        'matched-neural',
        str(config),
        '--output',
        'runs/offline',
        '--cohort',
        str(tmp_path / 'cohort.json'),
        '--parent-schedule',
        str(tmp_path / 'parent-schedule.json'),
        '--input-schedule',
        str(tmp_path / 'input-schedule.json'),
        '--input-attempt',
        str(tmp_path / 'input-attempt'),
        '--membership',
        str(tmp_path / 'membership.json'),
        '--schedule',
        str(tmp_path / 'schedule.json'),
        '--json',
    ]
    result = runner.invoke(app=app, args=args)
    assert result.exit_code == 0
    assert json.loads(s=result.stdout) == {'gate_passed': False}

"""Offline T33 memory comparison and CLI checks."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics import rate_access
from flystate.diagnostics.artifacts import attempt
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.storage.parquet import write_table

IDENTITIES: int = 20
PHOTOS: int = 10
POPULATION: str = 'central_brain'
REPRESENTATION: str = 'last_window_state'
DECODE: Path = Path('runs/offline/decode')
JSON_OPTION: str = '--json'
OK_RESULT: dict[str, bool] = {'ok': True}
OUTPUT_OPTION: str = '--output'
DIAGNOSE: str = 'diagnose'
RECORD_COMMAND: str = 'rate-record'
MEMORY_COMMAND: str = 'rate-memory'
GOOD: str = 'good'
FLAT: str = 'flat'
SUPPORTED: str = 'memory_supported'
GAIN_KEY: str = 'persistent_minus_reset_pp'
REPRESENTATION_KEY: str = 'representation'


def _rows(condition: str, correct: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 200 OOF rows in which the first ``correct`` photographs of each identity are right.

    :param condition: Recording name.
    :type condition: str
    :param correct: Correct photographs per identity, 0 to 10.
    :type correct: int
    :returns: OOF rows and the matching metrics row.
    :rtype: tuple[list[dict[str, Any]], dict[str, Any]]
    """
    rows = [
        {
            REPRESENTATION_KEY: f'{condition}/{POPULATION}/{REPRESENTATION}',
            'sample_id': f'id{label}-{rank}',
            'y_true': label,
            'y_pred': label if rank < correct else (label + 1) % IDENTITIES,
        }
        for label in range(IDENTITIES)
        for rank in range(PHOTOS)
    ]
    metric = {
        'condition': condition,
        'population': POPULATION,
        REPRESENTATION_KEY: REPRESENTATION,
        'top1': correct / PHOTOS,
        'log_loss': 1.0,
    }
    return rows, metric


def test_memory_gate_needs_a_clear_paired_gain(tiny_experiment: ExperimentConfig) -> None:
    """Pass a setting where persistent state wins by 40 points and fail one with no gain.

    :param tiny_experiment: Offline synthetic experiment configuration.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for condition, correct in (
        (f'{GOOD}-persistent', 6),
        (f'{GOOD}-reset', 2),
        (f'{FLAT}-persistent', 3),
        (f'{FLAT}-reset', 3),
    ):
        condition_rows, metric = _rows(condition=condition, correct=correct)
        rows.extend(condition_rows)
        metrics.append(metric)
    with attempt(paths=paths, cfg=tiny_experiment, output=DECODE, parameters={}) as directory:
        write_table(path=directory / 'oof-predictions.parquet', rows=rows)
        write_json(path=directory / 'report.json', value={'metrics': metrics})
    gate = rate_access.analyze_memory(
        cfg=tiny_experiment,
        paths=paths,
        output=Path('runs/offline/memory'),
        decodes=[DECODE],
        settings=[GOOD, FLAT],
        populations=[POPULATION],
        representation=REPRESENTATION,
        minimum_gain_pp=10.0,
    )
    assert gate == {POPULATION: True}
    report = json.loads(s=(paths.home / 'runs/offline/memory/report.json').read_text())
    by_setting = {row['setting']: row for row in report['comparisons']}
    assert by_setting[GOOD][GAIN_KEY] == pytest.approx(40.0)
    assert by_setting[GOOD][SUPPORTED]
    assert by_setting[FLAT][GAIN_KEY] == 0
    assert not by_setting[FLAT][SUPPORTED]
    assert report['bonferroni_level'] == pytest.approx(0.025)


def test_rate_cli_help_and_json(
    tiny_experiment: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose both T33 commands with help and exactly one JSON object on stdout.

    :param tiny_experiment: Offline valid configuration.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Isolated configuration file location.
    :type tmp_path: Path
    :param monkeypatch: Scoped implementation replacement.
    :type monkeypatch: pytest.MonkeyPatch
    """
    config = tmp_path / 'config.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    runner = CliRunner()
    for command in (RECORD_COMMAND, MEMORY_COMMAND):
        result = runner.invoke(app=app, args=[DIAGNOSE, command, '--help'])
        assert result.exit_code == 0
        assert JSON_OPTION in re.sub(pattern=r'\x1b\[[0-9;]*m', repl='', string=result.stdout)

    def fake(**kwargs: Any) -> dict[str, Any]:
        """Return a minimal result after checking the forwarded configuration.

        :param kwargs: Forwarded CLI inputs.
        :type kwargs: Any
        :returns: Synthetic result object.
        :rtype: dict[str, Any]
        """
        assert kwargs['cfg'].name == tiny_experiment.name
        return OK_RESULT

    monkeypatch.setattr('flystate.cli.rate_access.record_rate', fake)
    monkeypatch.setattr('flystate.cli.rate_access.analyze_memory', fake)
    record = runner.invoke(
        app=app,
        args=[
            DIAGNOSE,
            RECORD_COMMAND,
            str(config),
            OUTPUT_OPTION,
            'runs/r',
            '--cohort',
            'c.json',
            '--parent-schedule',
            'p.json',
            '--membership',
            'm.json',
            '--gain',
            '1',
            '--leak',
            '0.25',
            '--input-scale',
            '20',
            '--steps-per-window',
            '4',
            '--reset-each-window',
            JSON_OPTION,
        ],
    )
    assert record.exit_code == 0 and json.loads(s=record.stdout) == OK_RESULT
    memory = runner.invoke(
        app=app,
        args=[
            DIAGNOSE,
            MEMORY_COMMAND,
            str(config),
            str(DECODE),
            OUTPUT_OPTION,
            'runs/m',
            '--setting',
            'g1',
            '--population',
            POPULATION,
            JSON_OPTION,
        ],
    )
    assert memory.exit_code == 0 and json.loads(s=memory.stdout) == OK_RESULT

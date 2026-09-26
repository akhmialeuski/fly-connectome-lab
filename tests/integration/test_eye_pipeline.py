"""Offline T39 pipeline: eye sampling, evaluation and the frozen decision."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics.eye import LIMITED, NOT_VIABLE, PIXEL_REFERENCE, VIABLE, eye_cases
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths

RUNNER = CliRunner()
RECORD: str = 'runs/eye/record'
EVALUATE: str = 'runs/eye/evaluate'
OUTPUT: str = '--output'
DIAGNOSE: str = 'diagnose'
DECISION: str = 'decision'
PRIMARY_90: str = 'hex_luma_w90'


def _invoke(arguments: list[str]) -> dict[str, Any]:
    """Run one CLI command with ``--json`` and return its JSON object.

    :param arguments: Arguments after ``flystate``.
    :type arguments: list[str]
    :returns: Parsed stdout object.
    :rtype: dict[str, Any]
    """
    result = RUNNER.invoke(app, [*arguments, '--json'])
    assert result.exit_code == 0, result.output
    return json.loads(s=result.stdout)


def test_eye_pipeline_records_evaluates_and_decides(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Record every case, score them with the standard readout and apply the frozen rule.

    :param tiny_experiment: Offline synthetic experiment.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Test-owned directory for the YAML file.
    :type tmp_path: Path
    """
    config = tmp_path / 'tiny.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    recorded = _invoke([DIAGNOSE, 'eye-record', str(config), OUTPUT, RECORD])
    counts = recorded['ommatidia_per_case']
    assert counts['hex_luma_w120'] > counts[PRIMARY_90] > counts['hex_luma_w30']
    assert abs(counts['square_luma_w90'] - counts[PRIMARY_90]) <= 2 * np.sqrt(counts[PRIMARY_90])
    with np.load(get_paths().home / RECORD / 'responses.npz') as arrays:
        assert arrays['final_hex_rgb_w60'].shape[1] == 3 * counts['hex_rgb_w60']
    evaluated = _invoke(
        [DIAGNOSE, 'eye-evaluate', str(config), OUTPUT, EVALUATE, '--recording', RECORD]
    )
    assert set(evaluated['scores']) >= {f'eye/{name}' for name in [PIXEL_REFERENCE, *eye_cases()]}
    result = RUNNER.invoke(
        app,
        [
            DIAGNOSE,
            'eye-decide',
            str(config),
            OUTPUT,
            'runs/eye/decide',
            '--evaluation',
            EVALUATE,
            '--json',
        ],
    )
    reference = evaluated['scores'][f'eye/{PIXEL_REFERENCE}']
    if result.exit_code == 0:
        decided = json.loads(s=result.stdout)
        fraction = decided['primary_retained_fraction']
        assert decided[DECISION] == (
            VIABLE if fraction >= 0.5 else LIMITED if fraction >= 0.25 else NOT_VIABLE
        )
    else:
        # Synthetic faces can leave the pixel reference at chance, which the rule must report.
        assert 'not above chance' in result.output, (reference, result.output)

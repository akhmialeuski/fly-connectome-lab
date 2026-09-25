"""Offline end-to-end T37 wiring study on a synthetic brain and synthetic faces."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics.wiring import (
    ADVANTAGE,
    DISADVANTAGE,
    EQUIVALENT,
    INCONCLUSIVE,
    build_reservoir,
    classify,
    record_wiring,
)
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths
from flystate.storage.parquet import read_table

RUNNER = CliRunner()
COHORTS: tuple[int, ...] = (0, 1)
CENTRAL: str = 'central_brain'
MODEL: tuple[str, ...] = (
    '--leak',
    '0.25',
    '--driven-leak',
    '1.0',
    '--input-scale',
    '20',
    '--steps-per-window',
    '2',
    '--threads',
    '1',
)
ALPHA: str = '0.5'
RECORD_ROOT: str = 'runs/wiring/record'
REPORT: str = 'report.json'
PARAMETERS: str = 'parameters'
JSON: str = '--json'
OUTPUT: str = '--output'
COHORT: str = '--cohort'
FAMILY: str = '--family'
DIAGNOSE: str = 'diagnose'
FLY: str = 'fly'
ALPHA_OPTION: str = '--alpha'
CANDIDATE_OPTION: str = '--candidate'
CONTRAST_OPTION: str = '--contrast'
EVALUATION_OPTION: str = '--evaluation'
GRAPH_SEED_OPTION: str = '--graph-seed'
COHORTS_KEY: str = 'cohorts'
CORRECT: str = 'correct'
DEGREE: str = 'degree'
DEGREE0: str = 'degree0'
DIFFERENCE: str = 'difference_pp'
FEEDFORWARD: str = 'feedforward'
GRAPH: str = 'graph'
MCNEMAR: str = 'mcnemar'
RESET: str = 'reset'


def _invoke(arguments: list[str]) -> dict[str, Any]:
    """Run one CLI command with ``--json`` and return its single JSON object.

    :param arguments: Command-line arguments after ``flystate``.
    :type arguments: list[str]
    :returns: Parsed stdout object.
    :rtype: dict[str, Any]
    """
    result = RUNNER.invoke(app, [*arguments, JSON])
    assert result.exit_code == 0, result.output
    return json.loads(s=result.stdout)


def _record(config: Path, name: str, graph: list[str]) -> dict[str, Any]:
    """Record both synthetic cohorts on one graph.

    :param config: Base experiment YAML.
    :type config: Path
    :param name: Recording name.
    :type name: str
    :param graph: Family, seed and operating-point options.
    :type graph: list[str]
    :returns: Command result.
    :rtype: dict[str, Any]
    """
    cohorts = [item for seed in COHORTS for item in (COHORT, str(seed))]
    return _invoke(
        [
            DIAGNOSE,
            'wiring-record',
            str(config),
            OUTPUT,
            f'{RECORD_ROOT}/{name}',
            *cohorts,
            *graph,
            *MODEL,
        ]
    )


def test_wiring_study_records_selects_and_analyzes(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Run the full T37 pipeline and check gains, resumption, readout replay and contrasts.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Test-owned directory for the YAML file.
    :type tmp_path: Path
    """
    paths = get_paths()
    config = tmp_path / 'tiny.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    graphs = {
        FLY: [FAMILY, FLY, ALPHA_OPTION, ALPHA],
        RESET: [FAMILY, FLY, ALPHA_OPTION, ALPHA, '--reset-each-window'],
        DEGREE0: [FAMILY, DEGREE, GRAPH_SEED_OPTION, '0', ALPHA_OPTION, ALPHA],
        'degree1': [FAMILY, DEGREE, GRAPH_SEED_OPTION, '1', ALPHA_OPTION, ALPHA],
        'random0': [FAMILY, 'random_target', GRAPH_SEED_OPTION, '0', ALPHA_OPTION, ALPHA],
        FEEDFORWARD: [FAMILY, FEEDFORWARD, '--gain', '1.0'],
    }
    radii = {}
    for name, graph in graphs.items():
        result = _record(config=config, name=name, graph=graph)
        assert set(result[COHORTS_KEY].values()) == {'recorded'}
        radii[name] = result[GRAPH]['giant_component_radius']
        if name != FEEDFORWARD:
            assert np.isclose(result[GRAPH]['gain'] * radii[name], float(ALPHA))
    assert radii[FLY] == radii[RESET] != radii[DEGREE0]
    again = _record(config=config, name=FLY, graph=graphs[FLY])
    assert set(again[COHORTS_KEY].values()) == {'existing'}

    evaluations = []
    for seed in COHORTS:
        output = f'runs/wiring/evaluate/s{seed}'
        recordings = [f'{name}={RECORD_ROOT}/{name}/s{seed}' for name in graphs]
        _invoke(
            [
                DIAGNOSE,
                'wiring-evaluate',
                str(config),
                OUTPUT,
                output,
                COHORT,
                str(seed),
                *[item for recording in recordings for item in ('--recording', recording)],
            ]
        )
        evaluations.append(f's{seed}={output}')
        rows = read_table(path=paths.home / output / 'predictions.parquet')
        states = np.load(file=paths.home / RECORD_ROOT / FLY / f's{seed}' / 'responses.npz')
        report = json.loads(s=(paths.home / RECORD_ROOT / FLY / f's{seed}' / REPORT).read_text())
        order = {sample: index for index, sample in enumerate(report['sample_ids'])}
        model = np.load(file=paths.home / output / 'models' / FLY / CENTRAL / 'weights.npz')
        chosen = [row for row in rows if row['case'] == f'{FLY}/{CENTRAL}']
        x = states[f'final_{CENTRAL}'][[order[row['sample_id']] for row in chosen]]
        scores = x.astype(np.float64) @ model['weights'].T + model['bias']
        replayed = model['classes'][scores.argmax(axis=1)]
        assert np.array_equal(replayed, [row['predicted'] for row in chosen])

    selection = _invoke(
        [
            DIAGNOSE,
            'wiring-select',
            str(config),
            OUTPUT,
            'runs/wiring/select',
            COHORT,
            '0',
            EVALUATION_OPTION,
            'runs/wiring/evaluate/s0',
            CANDIDATE_OPTION,
            f'{FLY}:{ALPHA}={FLY}/{CENTRAL}',
            CANDIDATE_OPTION,
            f'degree:{ALPHA}=degree0/{CENTRAL}',
        ]
    )
    assert selection['families'][FLY]['selected_alpha'] == float(ALPHA)

    analysis = _invoke(
        [
            DIAGNOSE,
            'wiring-analyze',
            str(config),
            OUTPUT,
            'runs/wiring/analyze',
            *[item for evaluation in evaluations for item in (EVALUATION_OPTION, evaluation)],
            CONTRAST_OPTION,
            f'wiring={FLY}/{CENTRAL}:degree0/{CENTRAL},degree1/{CENTRAL}',
            CONTRAST_OPTION,
            f'memory={FLY}/{CENTRAL}:reset/{CENTRAL}',
        ]
    )
    accuracy = analysis['accuracy']
    photographs = analysis['photographs']
    assert photographs == 2 * 12
    wiring, memory = analysis['contrasts']
    expected = (
        accuracy[f'{FLY}/{CENTRAL}'][CORRECT]
        - (accuracy[f'degree0/{CENTRAL}'][CORRECT] + accuracy[f'degree1/{CENTRAL}'][CORRECT]) / 2
    )
    assert np.isclose(wiring[DIFFERENCE], 100 * expected / photographs)
    low, high = wiring['interval_95_pp']
    assert low <= wiring[DIFFERENCE] <= high
    assert MCNEMAR in memory
    assert MCNEMAR not in wiring
    assert set(wiring['per_cohort_pp']) == {'s0', 's1'}


def test_record_wiring_rejects_invalid_requests(tiny_experiment: ExperimentConfig) -> None:
    """Refuse cohorts that differ beyond their seed, ambiguous gains and misplaced seeds.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    other = tiny_experiment.model_copy(
        update={'readout': tiny_experiment.readout.model_copy(update={'pca_components': 5})}
    )
    common: dict[str, Any] = {
        'paths': paths,
        'output': Path('runs/wiring/invalid'),
        'seed': None,
        'leak': 0.25,
        'driven_leak': 1.0,
        'input_scale': 20.0,
        'steps_per_window': 2,
        'reset_each_window': False,
        'threads': 1,
    }
    with pytest.raises(expected_exception=ValueError, match='differ only'):
        record_wiring(
            cohorts={'a': tiny_experiment, 'b': other},
            family=FLY,
            alpha=0.5,
            gain=None,
            **common,
        )
    with pytest.raises(expected_exception=ValueError, match='exactly one'):
        record_wiring(cohorts={'a': tiny_experiment}, family=FLY, alpha=0.5, gain=1.0, **common)
    with pytest.raises(expected_exception=ValueError, match='seed given outside'):
        build_reservoir(
            paths=paths,
            family=DEGREE,
            seed=None,
            input_idx=np.arange(3, dtype=np.int64),
            leak=0.25,
            driven_leak=1.0,
            batch_size=1,
        )


@pytest.mark.parametrize(
    argnames=('low', 'high', 'expected'),
    argvalues=[
        (0.5, 3.0, ADVANTAGE),
        (-3.0, -0.5, DISADVANTAGE),
        (-4.9, 4.9, EQUIVALENT),
        (-6.0, 2.0, INCONCLUSIVE),
    ],
)
def test_classify_reads_intervals_against_the_margin(
    low: float, high: float, expected: str
) -> None:
    """Map each interval to the frozen reading with the 5-point margin.

    :param low: Lower bound, percentage points.
    :type low: float
    :param high: Upper bound, percentage points.
    :type high: float
    :param expected: Expected reading.
    :type expected: str
    """
    assert classify(low=low, high=high) == expected

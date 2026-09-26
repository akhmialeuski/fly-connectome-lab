"""Offline T38 recording in batches with blank delays, evaluation and pooled analysis."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from flystate.brain.rate import RateReservoir
from flystate.cli.main import app
from flystate.datasets.preprocess import prepare_dataset
from flystate.diagnostics.confirmation import driven_neurons
from flystate.diagnostics.drive_sweep import population_indices
from flystate.diagnostics.input_access import _features
from flystate.diagnostics.rate_access import simulate_states
from flystate.diagnostics.scale import half_retention, interference_partners, record_scale
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths

RUNNER = CliRunner()
MODEL: dict[str, Any] = {
    'gain': 1.0,
    'leak': 0.25,
    'driven_leak': 1.0,
    'input_scale': 20.0,
    'steps_per_window': 2,
    'threads': 1,
}
DELAYS: tuple[int, ...] = (0, 1, 3)
CENTRAL: str = 'central_brain'
FINAL_CENTRAL: str = f'final_{CENTRAL}'
FINAL_DELAY3: str = f'{FINAL_CENTRAL}_delay3'
ACCURACY: str = 'accuracy_pct'
CURVE: str = 'forgetting_curve_pct'
PERSISTENT_RUN: str = 'runs/scale/p'
RESET_RUN: str = 'runs/scale/r'
EVALUATION_RUN: str = 'runs/scale/e'
INTERFERENCE_RUN: str = 'runs/scale/i'
RECORD: str = 'scale-record'
RESPONSES: str = 'responses.npz'
JSON: str = '--json'
OUTPUT: str = '--output'
DELAY: str = '--delay'
DIAGNOSE: str = 'diagnose'
MODEL_OPTIONS: tuple[str, ...] = (
    '--gain',
    '1.0',
    '--leak',
    '0.25',
    '--driven-leak',
    '1.0',
    '--input-scale',
    '20',
    '--steps-per-window',
    '2',
    '--batch-size',
    '20',
    '--threads',
    '1',
)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    """Load every array of a recorded responses file.

    :param path: Attempt directory.
    :type path: Path
    :returns: Array name to values.
    :rtype: dict[str, np.ndarray]
    """
    with np.load(file=path / RESPONSES, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_batches_and_delays_match_one_direct_simulation(tiny_experiment: ExperimentConfig) -> None:
    """Give bit-identical states for any batch size, taken at the right blank windows.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    outputs = {}
    for batch in (40, 10):
        output = Path(f'runs/scale/batch{batch}')
        record_scale(
            cfg=tiny_experiment,
            paths=paths,
            output=output,
            reset_each_window=False,
            delays=DELAYS,
            batch_size=batch,
            **MODEL,
        )
        outputs[batch] = _arrays(path=paths.home / output)
    assert outputs[40].keys() == outputs[10].keys()
    assert all(np.array_equal(outputs[40][key], outputs[10][key]) for key in outputs[40])
    assert {FINAL_CENTRAL, f'{FINAL_CENTRAL}_delay1', FINAL_DELAY3} <= set(outputs[40])

    prepared = prepare_dataset(cfg=tiny_experiment, paths=paths)
    rows = list(range(len(prepared.samples)))
    features, _ = _features(cfg=tiny_experiment, paths=paths, prepared=prepared, rows=rows)
    glimpses = tiny_experiment.episodes.steps
    currents = features['encoded_current'].reshape(len(rows), glimpses, -1)
    input_idx = driven_neurons(cfg=tiny_experiment, paths=paths)
    reservoir = RateReservoir(
        brain_dir=paths.brain,
        gain=MODEL['gain'],
        leak=MODEL['leak'],
        batch_size=len(rows),
        leak_overrides={MODEL['driven_leak']: input_idx},
    )
    drive = np.zeros(shape=(len(rows), glimpses + max(DELAYS), currents.shape[2]), dtype=np.float32)
    drive[:, :glimpses] = currents * np.float32(MODEL['input_scale'])
    central = population_indices(brain_file=paths.brain / 'brain.npz', input_idx=input_idx)[CENTRAL]
    states = simulate_states(
        reservoir=reservoir,
        input_idx=input_idx,
        inputs=drive,
        populations={CENTRAL: central},
        steps_per_window=MODEL['steps_per_window'],
        reset_each_window=False,
    )[f'state_{CENTRAL}']
    assert np.array_equal(outputs[40][FINAL_CENTRAL], states[:, glimpses - 1])
    assert np.array_equal(outputs[40][FINAL_DELAY3], states[:, glimpses + 2])
    assert not np.array_equal(states[:, glimpses - 1], states[:, glimpses + 2])


def test_interference_shows_partner_glimpses(tiny_experiment: ExperimentConfig) -> None:
    """Pair photographs within their split across identities, and feed the partner's glimpses.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    samples = prepare_dataset(cfg=tiny_experiment, paths=paths).samples
    partners = interference_partners(samples=samples, seed=tiny_experiment.seed)
    assert np.array_equal(
        partners, interference_partners(samples=samples, seed=tiny_experiment.seed)
    )
    for row, partner in enumerate(partners):
        assert samples[partner].split == samples[row].split
        assert samples[partner].label != samples[row].label
    output = Path('runs/scale/interference')
    record_scale(
        cfg=tiny_experiment,
        paths=paths,
        output=output,
        reset_each_window=False,
        delays=DELAYS,
        batch_size=20,
        interference=True,
        **MODEL,
    )
    recorded = _arrays(path=paths.home / output)
    rows = list(range(len(samples)))
    features, _ = _features(
        cfg=tiny_experiment,
        paths=paths,
        prepared=prepare_dataset(cfg=tiny_experiment, paths=paths),
        rows=rows,
    )
    glimpses = tiny_experiment.episodes.steps
    currents = features['encoded_current'].reshape(len(rows), glimpses, -1)
    input_idx = driven_neurons(cfg=tiny_experiment, paths=paths)
    reservoir = RateReservoir(
        brain_dir=paths.brain,
        gain=MODEL['gain'],
        leak=MODEL['leak'],
        batch_size=len(rows),
        leak_overrides={MODEL['driven_leak']: input_idx},
    )
    drive = np.concatenate([currents, currents[partners, : max(DELAYS)]], axis=1)
    drive = (drive * np.float32(MODEL['input_scale'])).astype(np.float32)
    central = population_indices(brain_file=paths.brain / 'brain.npz', input_idx=input_idx)[CENTRAL]
    states = simulate_states(
        reservoir=reservoir,
        input_idx=input_idx,
        inputs=drive,
        populations={CENTRAL: central},
        steps_per_window=MODEL['steps_per_window'],
        reset_each_window=False,
    )[f'state_{CENTRAL}']
    assert np.array_equal(recorded[FINAL_CENTRAL], states[:, glimpses - 1])
    assert np.array_equal(recorded[FINAL_DELAY3], states[:, glimpses + 2])


def test_record_scale_rejects_invalid_requests(tiny_experiment: ExperimentConfig) -> None:
    """Refuse blank delays with a reset state and a batch that does not divide the cohort.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    with pytest.raises(expected_exception=ValueError, match='persistent state'):
        record_scale(
            cfg=tiny_experiment,
            paths=paths,
            output=Path('runs/scale/reset-delay'),
            reset_each_window=True,
            delays=DELAYS,
            batch_size=40,
            **MODEL,
        )
    with pytest.raises(expected_exception=ValueError, match='multiple of the reservoir batch'):
        record_scale(
            cfg=tiny_experiment,
            paths=paths,
            output=Path('runs/scale/odd-batch'),
            reset_each_window=False,
            delays=(0,),
            batch_size=7,
            **MODEL,
        )


def _invoke(arguments: list[str]) -> dict[str, Any]:
    """Run one CLI command with ``--json`` and return its single JSON object.

    :param arguments: Arguments after ``flystate``.
    :type arguments: list[str]
    :returns: Parsed stdout object.
    :rtype: dict[str, Any]
    """
    result = RUNNER.invoke(app, [*arguments, JSON])
    assert result.exit_code == 0, result.output
    return json.loads(s=result.stdout)


def test_scale_cli_records_evaluates_and_analyzes(
    tiny_experiment: ExperimentConfig, tmp_path: Path
) -> None:
    """Run the T38 commands end to end and read memory and forgetting from the analysis.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Test-owned directory for the YAML file.
    :type tmp_path: Path
    """
    config = tmp_path / 'tiny.yaml'
    config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
    delays = [item for delay in DELAYS for item in (DELAY, str(delay))]
    common = [str(config), '--set', 'encoder.seed=1']
    _invoke([DIAGNOSE, RECORD, *common, OUTPUT, PERSISTENT_RUN, *delays, *MODEL_OPTIONS])
    _invoke(
        [
            DIAGNOSE,
            RECORD,
            *common,
            OUTPUT,
            RESET_RUN,
            '--reset-each-window',
            *MODEL_OPTIONS,
        ]
    )
    _invoke(
        [
            DIAGNOSE,
            RECORD,
            *common,
            OUTPUT,
            INTERFERENCE_RUN,
            *delays,
            *MODEL_OPTIONS,
            '--interference',
        ]
    )
    _invoke(
        [
            DIAGNOSE,
            'scale-evaluate',
            *common,
            OUTPUT,
            EVALUATION_RUN,
            '--persistent-recording',
            PERSISTENT_RUN,
            '--reset-recording',
            RESET_RUN,
            '--interference-recording',
            INTERFERENCE_RUN,
            '--max-iterations',
            '60000',
            *delays,
        ]
    )
    evaluated = json.loads(s=(get_paths().home / EVALUATION_RUN / 'manifest.json').read_text())
    assert evaluated['parameters']['max_iterations'] == 60000
    recorded = json.loads(s=(get_paths().home / PERSISTENT_RUN / 'manifest.json').read_text())
    assert recorded['parameters']['encoder_seed'] == 1
    analysis = _invoke(
        [
            DIAGNOSE,
            'scale-analyze',
            str(config),
            OUTPUT,
            'runs/scale/a',
            '--development',
            f'seed1={EVALUATION_RUN}',
            '--confirmation',
            EVALUATION_RUN,
            *delays,
        ]
    )
    confirmation = analysis['confirmation']
    assert list(confirmation[CURVE]) == [str(delay) for delay in DELAYS]
    assert confirmation[CURVE]['0'] == confirmation[ACCURACY]
    assert list(confirmation['interference_curve_pct']) == [str(delay) for delay in DELAYS]
    assert confirmation['interference_curve_pct']['0'] == confirmation[ACCURACY]
    low_blank, high_blank = confirmation['blank_longest_delay_interval_95_pp']
    for prefix in ('blank', 'interference'):
        low, high = confirmation[f'{prefix}_longest_delay_interval_95_pp']
        assert low <= confirmation[f'{prefix}_longest_delay_change_pp'] <= high
    assert confirmation['blank_delays_preserve_identity'] == (low_blank > -5 and high_blank < 5)
    assert isinstance(analysis['decisions']['F1_blank_windows_preserve_identity'], bool)
    assert set(analysis['decisions']) == {
        'S1_recognition_at_scale',
        'S2_memory_at_scale',
        'F1_blank_windows_preserve_identity',
        'E_every_encoder_seed_keeps_memory',
    }
    assert analysis['development_spread'][ACCURACY]['standard_deviation'] is None


@pytest.mark.parametrize(
    argnames=('accuracies', 'expected'),
    argvalues=[
        ([0.51, 0.41, 0.31, 0.11], 2.5),
        ([0.51, 0.51, 0.46, 0.41], None),
    ],
)
def test_half_retention_interpolates_between_delays(
    accuracies: list[float], expected: float | None
) -> None:
    """Interpolate the delay at which accuracy above chance halves, or report that it never does.

    :param accuracies: Accuracy at delays 0, 1, 2 and 4, fractions.
    :type accuracies: list[float]
    :param expected: Expected half-retention delay, windows.
    :type expected: Optional[float]
    """
    result = half_retention(delays=[0, 1, 2, 4], accuracies=accuracies, chance=0.01)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)

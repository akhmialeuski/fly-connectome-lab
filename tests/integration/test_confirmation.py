"""Offline end-to-end T35 confirmation on a synthetic brain and synthetic faces."""

import json
from pathlib import Path

import numpy as np
import pytest

from flystate.diagnostics import confirmation
from flystate.diagnostics.artifacts import verify_attempt_inventory
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file
from flystate.settings import get_paths

PERSISTENT: str = 'persistent'
RESET: str = 'reset'
POPULATION: str = 'central_brain'


def test_confirmation_scores_every_held_out_photograph(tiny_experiment: ExperimentConfig) -> None:
    """Record persistent and reset states, then score only held-out photographs, paired.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    recordings = {}
    for name, reset in ((PERSISTENT, False), (RESET, True)):
        summary = confirmation.record_confirmation(
            cfg=tiny_experiment,
            paths=paths,
            output=Path(f'runs/offline/{name}'),
            gain=1.0,
            leak=0.25,
            driven_leak=1.0,
            input_scale=20.0,
            steps_per_window=2,
            reset_each_window=reset,
            shuffle_seed=None,
        )
        assert summary['parameters']['graph'] == 'fly'
        recordings[name] = Path(f'runs/offline/{name}')
    shuffled = confirmation.record_confirmation(
        cfg=tiny_experiment,
        paths=paths,
        output=Path('runs/offline/shuffled'),
        gain=1.0,
        leak=0.25,
        driven_leak=None,
        input_scale=20.0,
        steps_per_window=2,
        reset_each_window=False,
        shuffle_seed=0,
    )
    assert shuffled['parameters']['graph'] == 'degree_preserving_shuffle'
    result = confirmation.evaluate_confirmation(
        cfg=tiny_experiment,
        paths=paths,
        output=Path('runs/offline/evaluate'),
        recordings=recordings,
        populations=[POPULATION],
        comparisons=[(f'{PERSISTENT}/{POPULATION}', f'{RESET}/{POPULATION}')],
    )
    report = json.loads(s=(paths.home / 'runs/offline/evaluate/report.json').read_text())
    held_out = report['held_out_photographs']
    assert report['train_photographs'] + held_out == 40
    assert set(result['scores']) == {
        'input/encoded_current_all',
        'input/encoded_current_last',
        'input/pixels_all',
        f'{PERSISTENT}/{POPULATION}',
        f'{RESET}/{POPULATION}',
    }
    comparison = report['comparisons'][0]
    persistent = report['scores'][f'{PERSISTENT}/{POPULATION}']['held_out_correct']
    reset = report['scores'][f'{RESET}/{POPULATION}']['held_out_correct']
    assert np.isclose(comparison['a_minus_b_pp'], 100 * (persistent - reset) / held_out)
    assert report['scores'][f'{RESET}/{POPULATION}']['chance'] == 0.25
    assert (
        verify_attempt_inventory(directory=Path('runs/offline/evaluate'), paths=paths)['status']
        == 'completed'
    )
    for case, summary in report['scores'].items():
        model_dir = paths.home / 'runs/offline/evaluate/models' / case
        metadata = json.loads(s=(model_dir / 'model.json').read_text(encoding='utf-8'))
        assert metadata == summary['model']
        assert metadata['C'] == summary['C']
        assert metadata['weights_sha256'] == sha256_file(path=model_dir / 'weights.npz')
        with np.load(file=model_dir / 'weights.npz', allow_pickle=False) as arrays:
            assert {'scaler_mean', 'scaler_scale', 'coef', 'intercept', 'classes'} <= set(
                arrays.files
            )


def test_confirmation_rejects_recording_names_that_escape_model_directory(
    tiny_experiment: ExperimentConfig,
) -> None:
    """Reject an unsafe recording name before creating an evaluation attempt.

    :param tiny_experiment: Offline synthetic experiment configuration.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    output = Path('runs/offline/unsafe-evaluate')
    with pytest.raises(expected_exception=ValueError, match='Recording names'):
        confirmation.evaluate_confirmation(
            cfg=tiny_experiment,
            paths=paths,
            output=output,
            recordings={'../escape': Path('runs/offline/record')},
            populations=[POPULATION],
            comparisons=[],
        )
    assert not (paths.home / output).exists()

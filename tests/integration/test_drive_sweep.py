"""Offline T32 drive-sweep recording invariance, decoding, and CLI checks."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold
from typer.testing import CliRunner

from flystate.brain.runtime import EpisodeBrain
from flystate.cli.main import app
from flystate.diagnostics import drive_sweep
from flystate.diagnostics.artifacts import attempt
from flystate.diagnostics.temporal import _write_arrays
from flystate.encoders.sparse_projection import SparseProjectionEncoder
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.settings import get_paths
from flystate.storage.json import write_json

IDENTITIES: int = 20
FIT_PER_IDENTITY: int = 10
QUERY_PER_IDENTITY: int = 4
WINDOWS: int = 4
OUTPUT_OPTION: str = '--output'
COHORT_OPTION: str = '--cohort'
JSON_OPTION: str = '--json'
DIAGNOSE: str = 'diagnose'
RECORD_COMMAND: str = 'drive-record'
DECODE_COMMAND: str = 'drive-decode'
ALL_WINDOWS: str = 'all_windows_counts'
COHORT_SHA256: str = 'cohort_sha256'
FIT: str = 'fit'
FLAT: str = 'flat'
INFORMATIVE: str = 'informative'
LABEL: str = 'label'
ROLE: str = 'role'
SAMPLE_ID: str = 'sample_id'
PCA_COMPONENTS: str = 'pca_components'
OK_RESULT: dict[str, bool] = {'ok': True}
RECORD_OUTPUT: str = 'runs/r'


def _sample_id(label: int, rank: int) -> str:
    """Name one synthetic photograph.

    :param label: Identity label.
    :type label: int
    :param rank: Photograph rank within the identity.
    :type rank: int
    :returns: Stable synthetic sample identifier.
    :rtype: str
    """
    return f'id{label:02d}-{rank:02d}'


def test_populations_are_disjoint_and_exclude_driven(
    tiny_experiment: ExperimentConfig,
) -> None:
    """Keep every observed population disjoint from the driven neurons and from each other.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    brain = EpisodeBrain(
        brain_dir=paths.brain,
        brain_cfg=tiny_experiment.brain,
        readout_cfg=tiny_experiment.readout,
        batch_size=1,
        threads=1,
    )
    encoder = SparseProjectionEncoder(
        cfg=tiny_experiment.encoder,
        window=tiny_experiment.episodes.window,
        candidate_neurons=brain.cells(superclasses=['visual_projection']),
    )
    populations = drive_sweep.population_indices(
        brain_file=paths.brain / 'brain.npz', input_idx=encoder.input_idx
    )
    assert np.array_equal(populations['driven'], np.sort(encoder.input_idx))
    assert populations['kenyon'].size == 0 and populations['vnc'].size == 0
    members = np.concatenate(list(populations.values()))
    assert len(np.unique(members)) == len(members)
    assert populations['descending'].size == 100
    assert populations['visual_projection_other'].size == 600 - encoder.input_idx.size


def test_counts_do_not_depend_on_worker_partition(tiny_experiment: ExperimentConfig) -> None:
    """Reproduce identical arrays with one and two processes, and silence without drive or noise.

    :param tiny_experiment: Offline synthetic experiment with installed brain files.
    :type tiny_experiment: ExperimentConfig
    """
    paths = get_paths()
    rows = list(range(5))
    single, double = (
        drive_sweep.simulate_counts(
            cfg=tiny_experiment,
            paths=paths,
            rows=rows,
            amplitude_scale=4.0,
            noise_enabled=True,
            episode_seed=0,
            workers=workers,
        )
        for workers in (1, 2)
    )
    assert single.keys() == double.keys()
    for key, value in single.items():
        assert np.array_equal(value, double[key]), key
    driven = single['counts_driven']
    assert driven.shape == (5, tiny_experiment.episodes.steps, driven.shape[2])
    assert driven.sum() > 0
    silent = drive_sweep.simulate_counts(
        cfg=tiny_experiment,
        paths=paths,
        rows=rows[:2],
        amplitude_scale=0.0,
        noise_enabled=False,
        episode_seed=0,
        workers=1,
    )
    assert all(not value.any() for key, value in silent.items() if key.startswith('counts_')), (
        'A blank, noise-free episode must stay silent after a noise-free rest.'
    )


def _cohort_documents(directory: Path) -> tuple[Path, Path, Path, list[int]]:
    """Write a synthetic 280-photograph cohort, its exact folds, and a small T30-style schedule.

    :param directory: Test-owned document directory.
    :type directory: Path
    :returns: Cohort path, parent schedule path, selection schedule path, and labels.
    :rtype: tuple[Path, Path, Path, list[int]]
    """
    samples = []
    for label in range(IDENTITIES):
        for rank in range(FIT_PER_IDENTITY + QUERY_PER_IDENTITY):
            samples.append(
                {
                    SAMPLE_ID: _sample_id(label=label, rank=rank),
                    LABEL: label,
                    'rank': rank,
                    ROLE: FIT if rank < FIT_PER_IDENTITY else 'query',
                }
            )
    fit = [item for item in samples if item[ROLE] == FIT]
    fit_ids = [item[SAMPLE_ID] for item in fit]
    fit_labels = np.asarray([item[LABEL] for item in fit])
    folds = [
        {
            'index': index,
            'fit_ids': [fit_ids[i] for i in training],
            'held_out_ids': [fit_ids[i] for i in held_out],
            PCA_COMPONENTS: 10,
        }
        for index, (training, held_out) in enumerate(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(
                X=fit_labels, y=fit_labels
            )
        )
    ]
    cohort = directory / 'cohort.json'
    parent = directory / 'parent.json'
    selection = directory / 'selection.json'
    write_json(path=cohort, value={'samples': samples})
    write_json(path=parent, value={'cv': folds})
    write_json(
        path=selection,
        value={
            'candidate_C': [0.1],
            PCA_COMPONENTS: 10,
            'seed': 0,
            'tolerance': 1e-4,
            'max_iterations': 5000,
        },
    )
    return cohort, parent, selection, [item[LABEL] for item in samples]


def test_decode_scores_informative_population_above_flat_one(
    tiny_experiment: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decode only fit rows, score a label-carrying population high and a silent one at chance.

    :param tiny_experiment: Offline synthetic experiment configuration.
    :type tiny_experiment: ExperimentConfig
    :param tmp_path: Test-owned document directory.
    :type tmp_path: Path
    :param monkeypatch: Scoped replacement of source verification and input features.
    :type monkeypatch: pytest.MonkeyPatch
    """
    paths = get_paths()
    cohort, parent, selection, labels = _cohort_documents(directory=tmp_path)
    membership = tmp_path / 'membership.json'
    write_json(path=membership, value={})
    count = len(labels)
    rng = np.random.default_rng(seed=0)
    monkeypatch.setattr(target=drive_sweep, name='prepare_dataset', value=lambda **_: None)
    monkeypatch.setattr(
        target=drive_sweep, name='_verify_protocol', value=lambda **_: list(range(count))
    )
    monkeypatch.setattr(
        target=drive_sweep,
        name='_features',
        value=lambda **_: (
            {
                'encoded_current': rng.normal(size=(count, WINDOWS * 6)).astype(np.float32),
                'pixels': rng.normal(size=(count, WINDOWS * 6)).astype(np.float32),
            },
            {},
        ),
    )
    informative = np.zeros(shape=(count, WINDOWS, IDENTITIES), dtype=np.uint8)
    informative[np.arange(count), :, labels] = 5
    recording = Path('runs/offline/record')
    with attempt(
        paths=paths,
        cfg=tiny_experiment,
        output=recording,
        parameters={COHORT_SHA256: drive_sweep.sha256_file(path=cohort)},
    ) as directory:
        _write_arrays(
            path=directory / 'responses.npz',
            arrays={
                'labels': np.asarray(labels, dtype=np.int64),
                'counts_informative': informative,
                'final_voltage_informative': informative[:, -1].astype(np.float32),
                'counts_flat': np.zeros(shape=(count, WINDOWS, 8), dtype=np.uint8),
                'final_voltage_flat': np.zeros(shape=(count, 8), dtype=np.float32),
            },
        )
        write_json(
            path=directory / 'report.json',
            value={
                'parameters': {COHORT_SHA256: drive_sweep.sha256_file(path=cohort)},
                'sample_ids': [
                    _sample_id(label=label, rank=rank)
                    for label in range(IDENTITIES)
                    for rank in range(FIT_PER_IDENTITY + QUERY_PER_IDENTITY)
                ],
                'population_sizes': {INFORMATIVE: IDENTITIES, FLAT: 8},
            },
        )
    result = drive_sweep.decode_drive(
        cfg=tiny_experiment,
        paths=paths,
        output=Path('runs/offline/decode'),
        recordings=[recording],
        cohort_path=cohort,
        parent_schedule_path=parent,
        membership_path=membership,
        selection_schedule_path=selection,
    )
    assert result['cases'] == 4 + 6 and result['completed'] == 10
    report = json.loads(s=(paths.home / 'runs/offline/decode/report.json').read_text())
    scores = {
        (row['population'], row['representation']): row['top1_correct'] for row in report['metrics']
    }
    assert scores[(INFORMATIVE, ALL_WINDOWS)] == IDENTITIES * FIT_PER_IDENTITY
    assert scores[(FLAT, ALL_WINDOWS)] <= IDENTITIES
    assert not report['validation_test_reserve_scored'] and not report['prior_queries_scored']


def test_drive_cli_help_and_json(
    tiny_experiment: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose both commands with help text and exactly one JSON object on stdout.

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
    for command in (RECORD_COMMAND, DECODE_COMMAND):
        help_result = runner.invoke(app=app, args=[DIAGNOSE, command, '--help'])
        assert help_result.exit_code == 0
        visible = re.sub(pattern=r'\x1b\[[0-9;]*m', repl='', string=help_result.stdout)
        assert COHORT_OPTION in visible and JSON_OPTION in visible

    def fake(**kwargs: Any) -> dict[str, Any]:
        """Return a minimal result after checking the forwarded configuration.

        :param kwargs: Forwarded CLI inputs.
        :type kwargs: Any
        :returns: Synthetic result object.
        :rtype: dict[str, Any]
        """
        assert kwargs['cfg'].name == tiny_experiment.name
        return OK_RESULT

    monkeypatch.setattr('flystate.cli.drive_sweep.record_drive', fake)
    monkeypatch.setattr('flystate.cli.drive_sweep.decode_drive', fake)
    shared = [
        COHORT_OPTION,
        str(tmp_path / 'c.json'),
        '--parent-schedule',
        str(tmp_path / 'p.json'),
        '--membership',
        str(tmp_path / 'm.json'),
        JSON_OPTION,
    ]
    record = runner.invoke(
        app=app,
        args=[
            DIAGNOSE,
            RECORD_COMMAND,
            str(config),
            OUTPUT_OPTION,
            RECORD_OUTPUT,
            '--amplitude-scale',
            '4',
            '--no-noise',
            *shared,
        ],
    )
    assert record.exit_code == 0 and json.loads(s=record.stdout) == OK_RESULT
    decode = runner.invoke(
        app=app,
        args=[
            DIAGNOSE,
            DECODE_COMMAND,
            str(config),
            RECORD_OUTPUT,
            OUTPUT_OPTION,
            'runs/d',
            '--selection-schedule',
            str(tmp_path / 's.json'),
            *shared,
        ],
    )
    assert decode.exit_code == 0 and json.loads(s=decode.stdout) == OK_RESULT

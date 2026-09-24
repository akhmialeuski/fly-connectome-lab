"""Offline replay and blank/noise controls for recorded fly response trajectories."""

import hashlib
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.preprocess import prepare_dataset
from flystate.datasets.registry import require_validated
from flystate.experiments.config import ExperimentConfig, config_hash, effective_yaml
from flystate.hashing import sha256_file, sha256_obj, stable_int
from flystate.settings import Paths, get_paths
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table


def frozen_fixture_documents(
    cfg: ExperimentConfig, paths: Paths, directory: Path
) -> tuple[Path, Path, Path]:
    """Freeze two synthetic training images and matched anatomy without downloads.

    :param cfg: Small fixed experiment configuration.
    :type cfg: ExperimentConfig
    :param paths: Test-owned data home.
    :type paths: Paths
    :param directory: Test-owned protocol directory.
    :type directory: Path
    :returns: Cohort, masks, and source membership JSON paths.
    :rtype: tuple[Path, Path, Path]
    """
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    image_root = (
        CelebAAdapter(root=require_validated(paths=paths, name='celeba'), expected=None)
        .locate()
        .images_dir
    )
    ranked = sorted(
        (
            (row, sample)
            for row, sample in enumerate(prepared.samples)
            if sample.split == 'train' and sample.label == 0
        ),
        key=lambda pair: (stable_int(key=pair[1].sample_id), pair[1].sample_id),
    )[:2]
    members = {
        split: [sample.sample_id for sample in prepared.samples if sample.split == split]
        for split in ('train', 'val')
    }
    membership_path = directory / 'membership.json'
    write_json(
        path=membership_path,
        value={'sample_ids_by_split': members, 'membership_sha256': sha256_obj(obj=members)},
    )
    cohort_path = directory / 'cohort.json'
    write_json(
        path=cohort_path,
        value={
            'source_membership_sha256': sha256_obj(obj=members),
            'source_config_sha256': config_hash(cfg=cfg),
            'preprocess_cache_key': prepared.key,
            'windows': [0],
            'samples': [
                {
                    'sample_id': sample.sample_id,
                    'label': sample.label,
                    'source_sha256': sha256_file(path=image_root / sample.filename),
                    'aligned_sha256': hashlib.sha256(
                        string=prepared.images[row].tobytes()
                    ).hexdigest(),
                }
                for row, sample in ranked
            ],
        },
    )
    with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as metadata:
        ids = metadata['ids']
        superclass = metadata['superclass'].astype(str)
    masks = {}
    for name in ('visual_projection', 'descending_neuron'):
        candidates = np.flatnonzero(a=superclass == name)
        chosen = (
            candidates
            if len(candidates) == 100
            else np.sort(
                a=np.asarray(
                    a=sorted(
                        candidates,
                        key=lambda index: (stable_int(key=str(ids[index])), int(ids[index])),
                    )[:100],
                    dtype=np.int64,
                )
            )
        )
        body_ids = ids[chosen].astype(np.int64).tolist()
        masks[name] = {
            'full_count': len(candidates),
            'selected_count': len(chosen),
            'indices': chosen.tolist(),
            'body_ids': body_ids,
            'indices_sha256': sha256_obj(obj=chosen.tolist()),
            'body_ids_sha256': sha256_obj(obj=body_ids),
        }
    masks_path = directory / 'masks.json'
    write_json(
        path=masks_path,
        value={
            'brain_sha256': sha256_file(path=paths.brain / 'brain.npz'),
            'weights_sha256': sha256_file(path=paths.brain / 'weights.npz'),
            'masks': masks,
        },
    )
    return cohort_path, masks_path, membership_path


class TestTemporalPilot:
    """Keep case trajectories replayable and their paired controls physically aligned."""

    def test_cli_records_parity_and_paired_blanks(
        self, tiny_experiment: ExperimentConfig, tmp_path: Path
    ) -> None:
        """Replay corrected blanks and original stimuli through the public Typer CLI.

        :param tiny_experiment: Offline synthetic brain and CelebA-shaped dataset.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Test-owned protocol and config directory.
        :type tmp_path: Path
        """
        cfg = tiny_experiment.model_copy(
            update={
                'encoder': tiny_experiment.encoder.model_copy(update={'amplitude': 0.05}),
                'brain': tiny_experiment.brain.model_copy(
                    update={
                        'threads': 4,
                        'batch_size': 1,
                        'warmup_steps': 25,
                        'steps_per_observation': 10,
                    }
                ),
            }
        )
        paths = get_paths()
        config_path = tmp_path / 'temporal.yaml'
        config_path.write_text(data=effective_yaml(cfg=cfg), encoding='utf-8')
        cohort, masks, membership = frozen_fixture_documents(
            cfg=cfg, paths=paths, directory=tmp_path
        )
        runner = CliRunner()
        responses = {}
        checkpoints = {}
        for case in ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C0R', 'C6R'):
            output = Path(f'runs/diagnostics/temporal-fixture/{case}')
            result = runner.invoke(
                app,
                [
                    'diagnose',
                    'temporal',
                    str(config_path),
                    '--output',
                    str(output),
                    '--case',
                    case,
                    '--cohort',
                    str(cohort),
                    '--masks',
                    str(masks),
                    '--membership',
                    str(membership),
                    '--json',
                ],
            )
            assert result.exit_code == 0, result.output
            descriptor = json.loads(s=result.stdout)
            assert descriptor['status'] == 'completed'
            assert descriptor['sample_windows'] == 2
            checkpoints[case] = descriptor['checkpoints']
            response_file = paths.home / output / 'responses.npz'
            with np.load(file=response_file, allow_pickle=False) as source:
                responses[case] = {name: source[name].copy() for name in source.files}
        np.testing.assert_array_equal(
            actual=responses['C0']['noise_kicks'][:, :20],
            desired=responses['C1']['noise_kicks'],
        )
        assert np.any(responses['C1']['noise_kicks'])
        assert not np.any(responses['C5']['noise_kicks'])
        assert responses['C1']['voltage_descending_neuron'].shape == (2, 9, 100)
        assert responses['C1']['spikes_descending_neuron'].dtype == np.int32
        for original, corrected in (('C0', 'C0R'), ('C6', 'C6R')):
            positions = [checkpoints[corrected].index(step) for step in checkpoints[original]]
            for name, values in responses[original].items():
                expected = (
                    responses[corrected][name][:, positions]
                    if name.startswith(('voltage_', 'spikes_')) or name == 'readout_trace'
                    else responses[corrected][name]
                )
                np.testing.assert_array_equal(actual=values, desired=expected)
        assert {11, 12, 15}.issubset(checkpoints['C0R'])
        assert {11, 12, 15}.issubset(checkpoints['C6R'])
        source = Path('runs/diagnostics/temporal-fixture')
        analysis = Path('runs/diagnostics/temporal-fixture-analysis')
        result = runner.invoke(
            app,
            [
                'diagnose',
                'temporal-analyze',
                str(config_path),
                '--source',
                str(source),
                '--output',
                str(analysis),
                '--json',
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(s=result.stdout)
        assert report['status'] == 'completed'
        assert report['controls']['cases_verified'] == 9
        assert report['controls']['corrected_blank_overlap_exact']
        assert report['controls']['paired_noise_exact']
        assert report['controls']['noise_off_zero']
        assert report['rows'] == len(read_table(path=paths.home / analysis / 'per-unit.parquet'))
        assert report['decision'] in {'expand_training_only', 'inspect_encoder_and_recorder'}

    def test_cli_rejects_unknown_case(self) -> None:
        """Reject an unregistered intervention with CLI configuration exit code two."""
        runner = CliRunner()
        help_result = runner.invoke(app, ['diagnose', 'temporal', '--help'])
        assert help_result.exit_code == 0
        assert 'Record one frozen stimulus' in help_result.stdout
        assert '--json' in help_result.stdout
        analysis_help = runner.invoke(app, ['diagnose', 'temporal-analyze', '--help'])
        assert analysis_help.exit_code == 0
        assert '--source' in analysis_help.stdout
        assert '--json' in analysis_help.stdout
        result = runner.invoke(
            app,
            [
                'diagnose',
                'temporal',
                'unused.yaml',
                '--output',
                'runs/diagnostics/unused',
                '--case',
                'C7',
                '--cohort',
                'unused.json',
                '--masks',
                'unused.json',
                '--membership',
                'unused.json',
                '--json',
            ],
        )
        assert result.exit_code == 2
        assert 'error' in json.loads(s=result.stdout)

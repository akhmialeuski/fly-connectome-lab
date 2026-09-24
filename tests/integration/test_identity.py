"""Offline CLI replay of frozen identity responses and exact common noise."""

import hashlib
import json
from pathlib import Path

import numpy as np
from click import unstyle
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


def frozen_identity_fixture(
    cfg: ExperimentConfig, paths: Paths, directory: Path
) -> tuple[Path, Path, Path, Path]:
    """Freeze four synthetic identities and three observation windows offline.

    :param cfg: Small real-API experiment with 16 raster windows.
    :type cfg: ExperimentConfig
    :param paths: Test-owned brain and dataset home.
    :type paths: Paths
    :param directory: Test-owned protocol directory.
    :type directory: Path
    :returns: Cohort, masks, membership, and schedule document paths.
    :rtype: tuple[Path, Path, Path, Path]
    """
    prepared = prepare_dataset(cfg=cfg, paths=paths)
    image_root = (
        CelebAAdapter(root=require_validated(paths=paths, name='celeba'), expected=None)
        .locate()
        .images_dir
    )
    members = {
        split: [sample.sample_id for sample in prepared.samples if sample.split == split]
        for split in ('train', 'val')
    }
    membership = directory / 'membership.json'
    write_json(
        path=membership,
        value={'sample_ids_by_split': members, 'membership_sha256': sha256_obj(obj=members)},
    )
    with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as metadata:
        body_ids = metadata['ids']
        superclass = metadata['superclass'].astype(str)
    masks: dict[str, dict[str, object]] = {}
    names = ('visual_projection', 'cb_intrinsic', 'descending_neuron')
    for name in names:
        candidates = np.flatnonzero(a=superclass == name)
        ranked_indices = sorted(
            candidates,
            key=lambda index: (stable_int(key=str(body_ids[index])), int(body_ids[index])),
        )
        chosen = np.sort(a=np.asarray(a=ranked_indices[:20], dtype=np.int64))
        selected_ids = body_ids[chosen].astype(np.int64).tolist()
        masks[name] = {
            'full_count': len(candidates),
            'selected_count': len(chosen),
            'indices': chosen.tolist(),
            'body_ids': selected_ids,
            'indices_sha256': sha256_obj(obj=chosen.tolist()),
            'body_ids_sha256': sha256_obj(obj=selected_ids),
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
    selected = []
    for label in range(cfg.dataset.subset.n_identities):
        ranked_images = sorted(
            (
                (index, sample)
                for index, sample in enumerate(prepared.samples)
                if sample.split == 'train' and sample.label == label
            ),
            key=lambda pair: (stable_int(key=pair[1].sample_id), pair[1].sample_id),
        )[:2]
        for index, sample in ranked_images:
            selected.append(
                {
                    'sample_id': sample.sample_id,
                    'label': label,
                    'source_sha256': sha256_file(path=image_root / sample.filename),
                    'aligned_sha256': hashlib.sha256(prepared.images[index].tobytes()).hexdigest(),
                }
            )
    cohort = directory / 'cohort.json'
    write_json(
        path=cohort,
        value={
            'source_membership_sha256': sha256_obj(obj=members),
            'source_config_sha256': config_hash(cfg=cfg),
            'source_pilot_masks_sha256': sha256_file(path=masks_path),
            'preprocess_cache_key': prepared.key,
            'windows': [0, 7, 15],
            'samples': selected,
        },
    )
    schedule = directory / 'schedule.json'
    write_json(
        path=schedule,
        value={
            'schema_version': 1,
            'cases': [
                {'id': 'N0', 'noise_enabled': True, 'episode_seed': 0},
                {'id': 'N1', 'noise_enabled': True, 'episode_seed': 1},
                {'id': 'OFF', 'noise_enabled': False, 'episode_seed': 0},
            ],
            'cohort_sha256': sha256_file(path=cohort),
            'masks_sha256': sha256_file(path=masks_path),
            'membership_file_sha256': sha256_file(path=membership),
            'windows': [0, 7, 15],
            'checkpoints': [0, 1, 2, 5, 10, 11, 12, 15, 20],
            'stimulus_steps': 10,
            'recovery_steps': 10,
            'masks_order': list(names),
            'common_noise_identifier_template': 'identity-access|window=<window>|stream=<case>',
            'primary_endpoint_step': 10,
            'null_permutations': 10_000,
            'null_seed_key': 't28-primary-null',
            'memory_limit_bytes': 12 * 1024**3,
            'campaign_time_limit_seconds': 3600,
        },
    )
    return cohort, masks_path, membership, schedule


class TestIdentityAccess:
    """Keep image and noise identity separate while preserving fly trajectories."""

    def test_cli_records_common_noise_and_immutable_attempts(
        self, tiny_experiment: ExperimentConfig, tmp_path: Path
    ) -> None:
        """Replay all three conditions with exact blank noise-index fingerprints.

        :param tiny_experiment: Offline synthetic graph and CelebA-shaped source.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Test-owned protocol and attempts.
        :type tmp_path: Path
        """
        cfg = tiny_experiment.model_copy(
            update={
                'episodes': tiny_experiment.episodes.model_copy(update={'steps': 16}),
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
        config = tmp_path / 'identity.yaml'
        config.write_text(data=effective_yaml(cfg=cfg), encoding='utf-8')
        cohort, masks, membership, schedule = frozen_identity_fixture(
            cfg=cfg, paths=paths, directory=tmp_path
        )
        runner = CliRunner()
        help_result = runner.invoke(app, ['diagnose', 'identity', '--help'])
        assert help_result.exit_code == 0
        assert '--json' in unstyle(help_result.stdout)
        assert '--schedule' in unstyle(help_result.stdout)
        analysis_help = runner.invoke(app, ['diagnose', 'identity-analyze', '--help'])
        assert analysis_help.exit_code == 0
        assert '--source' in unstyle(analysis_help.stdout)
        assert '--json' in unstyle(analysis_help.stdout)
        digests = {}
        for case in ('N0', 'N1', 'OFF'):
            output = Path(f'runs/diagnostics/identity-fixture/{case}')
            args = [
                'diagnose',
                'identity',
                str(config),
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
                '--schedule',
                str(schedule),
                '--json',
            ]
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
            report = json.loads(s=result.stdout)
            assert report['status'] == 'completed'
            assert report['sample_windows'] == 24
            with (
                np.load(file=paths.home / output / 'responses.npz', allow_pickle=False) as driven,
                np.load(file=paths.home / output / 'blanks.npz', allow_pickle=False) as blank,
            ):
                assert driven['voltage_visual_projection'].shape == (24, 9, 20)
                assert driven['noise_digests'].shape == (24, 20, 32)
                for index in range(24):
                    window_slot = index % 3
                    np.testing.assert_array_equal(
                        actual=driven['noise_digests'][index],
                        desired=blank['noise_digests'][window_slot],
                    )
                digests[case] = driven['noise_digests'][0].copy()
                if case == 'OFF':
                    assert not np.any(driven['noise_kicks'])
                else:
                    assert np.any(driven['noise_kicks'])
            repeated = runner.invoke(app, args)
            assert repeated.exit_code == 1
        assert not np.array_equal(digests['N0'], digests['N1'])
        analysis = Path('runs/diagnostics/identity-fixture-analysis')
        analyzed = runner.invoke(
            app,
            [
                'diagnose',
                'identity-analyze',
                str(config),
                '--source',
                'runs/diagnostics/identity-fixture',
                '--output',
                str(analysis),
                '--cohort',
                str(cohort),
                '--masks',
                str(masks),
                '--membership',
                str(membership),
                '--schedule',
                str(schedule),
                '--json',
            ],
        )
        assert analyzed.exit_code == 0, analyzed.output
        analysis_report = json.loads(s=analyzed.stdout)
        assert analysis_report['controls']['cases_verified'] == 3
        assert analysis_report['controls']['paired_noise_indices_exact']
        assert analysis_report['controls']['input_digests_exact']
        assert analysis_report['primary']['N0']['queries'] == 8
        assert analysis_report['primary']['N0']['within_pairs'] == 4
        assert analysis_report['chance_accuracy'] == 0.25
        assert analysis_report['pair_rows'] == len(
            read_table(path=paths.home / analysis / 'pairs.parquet')
        )
        assert analysis_report['query_rows'] == len(
            read_table(path=paths.home / analysis / 'queries.parquet')
        )
        assert analysis_report['cross_seed_rows'] == len(
            read_table(path=paths.home / analysis / 'cross-seed.parquet')
        )
        unknown = runner.invoke(
            app,
            [
                'diagnose',
                'identity',
                'unused.yaml',
                '--output',
                'runs/diagnostics/unused/OTHER',
                '--case',
                'OTHER',
                '--cohort',
                str(cohort),
                '--masks',
                str(masks),
                '--membership',
                str(membership),
                '--schedule',
                str(schedule),
                '--json',
            ],
        )
        assert unknown.exit_code == 2
        assert 'error' in json.loads(s=unknown.stdout)

"""Offline replay and split-integrity checks for the matched noise study."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics import noise
from flystate.diagnostics.data import load_representation
from flystate.diagnostics.noise import generate_noise_trace, load_noise_trace
from flystate.diagnostics.noise_analysis import analyze_noise_recognition, identity_cluster_interval
from flystate.diagnostics.probes import run_probe
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_obj
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.traces.builder import build_trace, resolve_trace


class TestNoiseRecognition:
    """Verify exact baseline replay before comparing the isolated interventions."""

    def test_cli_replay_precision_and_probe(self, tiny_experiment: ExperimentConfig) -> None:
        """Freeze synthetic membership, replay native data, and fit the shared readout.

        :param tiny_experiment: Synthetic images, connectome, and fixed configuration.
        :type tiny_experiment: ExperimentConfig
        """
        cfg, paths = tiny_experiment, get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        groups = {
            split: sorted(s.sample_id for s in prepared.samples if s.split == split)
            for split in ('train', 'val')
        }
        membership = paths.home / 'membership.json'
        write_json(
            path=membership,
            value={
                'sample_ids_by_split': groups,
                'membership_sha256': sha256_obj(obj=groups),
            },
        )
        config = paths.home / 'config.yaml'
        config.write_text(data=effective_yaml(cfg=cfg), encoding='utf-8')
        runner = CliRunner()
        help_result = runner.invoke(app=app, args=['diagnose', 'noise-trace', '--help'])
        assert help_result.exit_code == 0
        assert help_result.stdout.strip()
        source = paths.runs / 'noise-baseline'
        result = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'noise-trace',
                str(config),
                '--membership',
                str(membership),
                '--output',
                str(source),
                '--episode-seed',
                str(cfg.seed),
                '--json',
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(s=result.stdout)['provenance']['native_replay_exact'] is True
        baseline = load_representation(
            cfg=cfg,
            paths=paths,
            representation='neural',
            history='last',
            features='both',
        )
        rounded = load_noise_trace(
            cfg=cfg,
            paths=paths,
            source=source,
            precision='float16',
            history='last',
            features='both',
        )
        np.testing.assert_array_equal(actual=rounded.x, desired=baseline.x)
        native = load_noise_trace(
            cfg=cfg,
            paths=paths,
            source=source,
            precision='float32',
            history='last',
            features='both',
        )
        assert native.x.dtype == np.float32
        assert len(native.samples) == len(groups['train']) + len(groups['val'])
        assert {s.split for s in native.samples} == {'train', 'val'}
        old = run_probe(
            cfg=cfg,
            paths=paths,
            output=paths.runs / 'old-probe',
            representation='neural',
            history='last',
            features='both',
            components=3,
            label_mode='true',
        )
        new = run_probe(
            cfg=cfg,
            paths=paths,
            output=paths.runs / 'new-probe',
            representation='neural',
            history='last',
            features='both',
            components=3,
            label_mode='true',
            trace_source=source,
            trace_precision='float16',
        )
        assert new['scores'] == old['scores']
        assert new['model']['weights_sha256'] == old['model']['weights_sha256']
        with pytest.raises(expected_exception=ValueError, match='inventory'):
            (source / 'checksums.sha256').write_text(data='', encoding='utf-8')
            load_noise_trace(
                cfg=cfg,
                paths=paths,
                source=source,
                precision='float16',
                history='last',
                features='both',
            )

    def test_controls_and_failed_replay(
        self, tiny_experiment: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Check noise-off warmup independence and retain a deliberately failed replay.

        :param tiny_experiment: Offline fixed source configuration.
        :type tiny_experiment: ExperimentConfig
        :param monkeypatch: Local response corruption for the replay gate.
        :type monkeypatch: pytest.MonkeyPatch
        """
        cfg, paths = tiny_experiment, get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        groups = {
            split: sorted(s.sample_id for s in prepared.samples if s.split == split)
            for split in ('train', 'val')
        }
        digest = sha256_obj(obj=groups)
        enabled = paths.runs / 'enabled'
        disabled = paths.runs / 'disabled'
        for output, on in ((enabled, True), (disabled, False)):
            generate_noise_trace(
                cfg=cfg,
                paths=paths,
                output=output,
                membership=groups,
                membership_sha256=digest,
                episode_seed=cfg.seed,
                noise_enabled=on,
            )
        with (
            np.load(file=enabled / 'rest.npz', allow_pickle=False) as first,
            np.load(file=disabled / 'rest.npz', allow_pickle=False) as second,
        ):
            np.testing.assert_array_equal(actual=first['voltage'], desired=second['voltage'])
            np.testing.assert_array_equal(actual=first['fired'], desired=second['fired'])
        with (
            np.load(file=enabled / 'features.npz', allow_pickle=False) as first,
            np.load(file=disabled / 'features.npz', allow_pickle=False) as second,
        ):
            assert not np.array_equal(a1=first['features'], a2=second['features'])
        original = noise.record_responses

        def corrupt(**kwargs: Any) -> tuple[NDArray, NDArray]:
            """Change one native feature after deterministic simulation.

            :param kwargs: Recording arguments forwarded without modification.
            :type kwargs: Any
            :returns: Corrupted features and unmodified spike counts.
            :rtype: tuple[NDArray, NDArray]
            """
            values, counts = original(**kwargs)
            values[0, 0, 0] += 1
            return values, counts

        monkeypatch.setattr(target=noise, name='record_responses', value=corrupt)
        failed = paths.runs / 'failed'
        with pytest.raises(expected_exception=ValueError, match='Native float16 replay'):
            generate_noise_trace(
                cfg=cfg,
                paths=paths,
                output=failed,
                membership=groups,
                membership_sha256=digest,
                episode_seed=cfg.seed,
                noise_enabled=True,
            )
        assert json.loads(s=(failed / 'manifest.json').read_text())['status'] == 'failed'
        assert (failed / 'features.npz').exists()

    def test_invalid_cli_options(self, tiny_experiment: ExperimentConfig, tmp_path: Path) -> None:
        """Reject malformed membership, unpaired precision, and an invalid analysis schedule.

        :param tiny_experiment: Fixed offline experiment configuration.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Isolated CLI configuration location.
        :type tmp_path: Path
        """
        config = tmp_path / 'config.yaml'
        config.write_text(data=effective_yaml(cfg=tiny_experiment), encoding='utf-8')
        membership = tmp_path / 'membership.json'
        write_json(path=membership, value={'sample_ids_by_split': {'train': [], 'val': []}})
        runner = CliRunner()
        result = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'noise-trace',
                str(config),
                '--membership',
                str(membership),
                '--output',
                'runs/invalid',
                '--episode-seed',
                '0',
                '--json',
            ],
        )
        assert result.exit_code == 2 and 'error' in json.loads(s=result.stdout)
        paired = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'run',
                str(config),
                '--output',
                'runs/invalid-probe',
                '--trace-source',
                'runs/source',
                '--json',
            ],
        )
        assert paired.exit_code == 2 and 'error' in json.loads(s=paired.stdout)
        assert runner.invoke(app=app, args=['diagnose', 'noise-analyze', '--help']).exit_code == 0
        schedule = tmp_path / 'schedule.json'
        write_json(path=schedule, value={})
        analysis = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'noise-analyze',
                str(config),
                '--schedule',
                str(schedule),
                '--output',
                'runs/invalid-analysis',
                '--json',
            ],
        )
        assert analysis.exit_code == 2 and 'error' in json.loads(s=analysis.stdout)

    def test_identity_cluster_interval(self) -> None:
        """Resample identities rather than treating their paired photographs as independent."""
        differences = np.asarray(a=[1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        identities = np.asarray(a=[0, 0, 1, 1])
        first = identity_cluster_interval(
            differences=differences, identities=identities, samples=2000, seed=0
        )
        second = identity_cluster_interval(
            differences=differences, identities=identities, samples=2000, seed=0
        )
        assert first == second
        assert first['clusters'] == 2
        assert first['diff_pp'] == 50.0
        assert first['ci_low_pp'] == 0.0 and first['ci_high_pp'] == 100.0
        with pytest.raises(expected_exception=ValueError, match='balanced'):
            identity_cluster_interval(
                differences=differences[:3], identities=identities[:3], samples=2000, seed=0
            )

    def test_complete_analysis_replays_prior_endpoint(
        self, tiny_experiment: ExperimentConfig
    ) -> None:
        """Fit six paired synthetic readouts and require exact historical baseline replay.

        :param tiny_experiment: Synthetic connectome and generated face-shaped source.
        :type tiny_experiment: ExperimentConfig
        """
        base = tiny_experiment
        cfg = base.model_copy(
            update={
                'dataset': base.dataset.model_copy(
                    update={
                        'subset': base.dataset.subset.model_copy(update={'images_per_identity': 20})
                    }
                ),
                'episodes': base.episodes.model_copy(update={'steps': 16}),
                'readout': base.readout.model_copy(
                    update={
                        'pca_components': 60,
                        'c_grid': (0.01, 0.1, 1.0, 10.0),
                        'cv_folds': 5,
                    }
                ),
            }
        )
        paths = get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        groups = {
            split: sorted(s.sample_id for s in prepared.samples if s.split == split)
            for split in ('train', 'val')
        }
        digest = sha256_obj(obj=groups)
        historical = paths.runs / 'historical'
        run_probe(
            cfg=cfg,
            paths=paths,
            output=historical,
            representation='neural',
            history='last',
            features='both',
            components=60,
            label_mode='true',
            max_iterations=50000,
            train_per_class=14,
            subset_seed=0,
        )
        schedule: dict[str, Any] = {
            'membership_sha256': digest,
            'historical': str(historical),
            'traces': {},
            'probes': {},
        }
        for case, seed, enabled in (
            ('seed0', 0, True),
            ('seed1', 1, True),
            ('off', 0, False),
        ):
            trace = paths.runs / f'trace-{case}'
            generate_noise_trace(
                cfg=cfg,
                paths=paths,
                output=trace,
                membership=groups,
                membership_sha256=digest,
                episode_seed=seed,
                noise_enabled=enabled,
            )
            schedule['traces'][case] = str(trace)
            schedule['probes'][case] = {}
            for precision in ('float32', 'float16'):
                output = paths.runs / f'probe-{case}-{precision}'
                run_probe(
                    cfg=cfg,
                    paths=paths,
                    output=output,
                    representation='neural',
                    history='last',
                    features='both',
                    components=60,
                    label_mode='true',
                    max_iterations=50000,
                    train_per_class=14,
                    subset_seed=0,
                    trace_source=trace,
                    trace_precision=precision,
                )
                schedule['probes'][case][precision] = str(output)
        result = analyze_noise_recognition(
            cfg=cfg, paths=paths, output=paths.runs / 'analysis', schedule=schedule
        )
        assert result['historical_replay']['exact_replay'] is True
        assert result['gate'] in {'promote_noise_off', 'do_not_promote'}
        assert set(result['precision_effects']) == {'seed0', 'seed1', 'off'}
        assert result['paired_validation']['float32']['mean_noisy']['clusters'] == 4
        assert result['paired_validation']['float16']['mean_noisy']['clusters'] == 4

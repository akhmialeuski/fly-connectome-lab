"""Synthetic paired-noise, native replay, ordering and immutable failure checks."""

import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.diagnostics import stability
from flystate.diagnostics.stability import response_distances, run_stability
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_obj
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.traces.builder import build_trace, resolve_trace


class TestSignalStability:
    """Keep noise interventions independent of images and preserve original trace semantics."""

    def test_distances(self) -> None:
        """Check analytic distances, missing pair groups, and invalid geometry."""
        first = np.asarray(a=[0.0, 1.0, 3.0]).reshape(3, 1, 1)
        report = response_distances(first=first, second=first + 1, labels=np.array([0, 0, 1]))
        assert report['same_image_different_noise']['mean_squared_distance_by_observation'] == [1]
        assert report['same_identity_different_image']['mean_squared_distance_by_observation'] == [
            1
        ]
        assert report['different_identity_same_noise']['mean_squared_distance_by_observation'] == [
            6.5
        ]
        one = response_distances(first=first[:1], second=first[:1], labels=np.array([0]))
        assert one['different_identity_same_noise']['pairs'] == 0
        assert one['different_identity_same_noise']['mean_squared_distance_by_observation'] is None
        for bad in (
            np.zeros(shape=(0, 1, 1)),
            np.array([1]),
            np.full(shape=(3, 1, 1), fill_value=np.nan),
        ):
            with pytest.raises(expected_exception=ValueError, match='matched finite'):
                response_distances(first=bad, second=first, labels=np.array([0, 0, 1]))

    def test_cli_and_reordering(self, tiny_experiment: ExperimentConfig) -> None:
        """Replay original features, run every control, and preserve responses after reordering.

        :param tiny_experiment: Offline deterministic images and synthetic connectome.
        :type tiny_experiment: ExperimentConfig
        """
        cfg = tiny_experiment
        paths = get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        ids = [s.sample_id for s in prepared.samples if s.split == 'train'][:4]
        membership = paths.home / 'membership.json'
        write_json(
            path=membership, value={'sample_ids': ids, 'sample_ids_sha256': sha256_obj(obj=ids)}
        )
        config = paths.home / 'config.yaml'
        config.write_text(data=effective_yaml(cfg=cfg))
        runner = CliRunner()
        assert runner.invoke(app=app, args=['diagnose', 'stability', '--help']).exit_code == 0
        output = paths.runs / 'stability'
        result = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'stability',
                str(config),
                '--membership',
                str(membership),
                '--output',
                str(output),
                '--json',
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads(s=result.stdout)
        assert report['native_replay_exact'] is True
        assert sum(c['episodes'] for c in report['cases']) == 4 * len(ids) + 3
        assert report['parameters']['trainable_parameters'] == []
        assert 'both' in report['distances']
        reordered = list(reversed(ids))
        second = paths.runs / 'reordered'
        run_stability(
            cfg=cfg,
            paths=paths,
            output=second,
            sample_ids=reordered,
            membership_sha256=sha256_obj(obj=reordered),
        )
        for case in ('native', 'matched-0', 'matched-1', 'noise-off'):
            with (
                np.load(file=output / f'{case}.npz', allow_pickle=False) as a,
                np.load(file=second / f'{case}.npz', allow_pickle=False) as b,
            ):
                np.testing.assert_array_equal(actual=a['features'], desired=b['features'][::-1])
        with (
            np.load(file=output / 'rest.npz', allow_pickle=False) as a,
            np.load(file=second / 'rest.npz', allow_pickle=False) as b,
        ):
            np.testing.assert_array_equal(actual=a['voltage'], desired=b['voltage'])
        write_json(path=membership, value={})
        invalid = runner.invoke(
            app=app,
            args=[
                'diagnose',
                'stability',
                str(config),
                '--membership',
                str(membership),
                '--output',
                str(output),
                '--json',
            ],
        )
        assert invalid.exit_code == 2 and 'error' in json.loads(s=invalid.stdout)
        assert (
            runner.invoke(
                app=app,
                args=[
                    'diagnose',
                    'stability',
                    str(config),
                    '--membership',
                    str(membership),
                    '--output',
                    str(output),
                ],
            ).exit_code
            == 2
        )

    def test_failures(
        self, tiny_experiment: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject invalid training membership and stop after a deliberately broken native replay.

        :param tiny_experiment: Offline baseline source configuration.
        :type tiny_experiment: ExperimentConfig
        :param monkeypatch: Isolated response corruption for the replay gate.
        :type monkeypatch: pytest.MonkeyPatch
        """
        cfg, paths = tiny_experiment, get_paths()
        build_trace(cfg=cfg, paths=paths)
        prepared, _, _ = resolve_trace(cfg=cfg, paths=paths)
        ids = [s.sample_id for s in prepared.samples if s.split == 'train'][:2]
        for index, invalid in enumerate(([], [ids[0], ids[0]], ['not-training'])):
            out = paths.runs / f'invalid-{index}'
            with pytest.raises(expected_exception=ValueError):
                run_stability(
                    cfg=cfg,
                    paths=paths,
                    output=out,
                    sample_ids=invalid,
                    membership_sha256=sha256_obj(obj=invalid),
                )
            assert json.loads(s=(out / 'manifest.json').read_text())['status'] == 'failed'
        original = stability.record_responses

        def corrupt(**kwargs: Any) -> tuple[NDArray, NDArray]:
            """Alter one native value to exercise the scientific replay gate.

            :param kwargs: Original recording arguments.
            :type kwargs: Any
            :returns: Corrupted features and untouched spike summaries.
            :rtype: tuple[NDArray, NDArray]
            """
            values, counts = original(**kwargs)
            values[0, 0, 0] += 1
            return values, counts

        monkeypatch.setattr(target=stability, name='record_responses', value=corrupt)
        out = paths.runs / 'broken-replay'
        with pytest.raises(expected_exception=ValueError, match='Native float16 replay'):
            run_stability(
                cfg=cfg,
                paths=paths,
                output=out,
                sample_ids=ids,
                membership_sha256=sha256_obj(obj=ids),
            )
        assert (out / 'native.npz').exists() and not (out / 'matched-0.npz').exists()
        assert json.loads(s=(out / 'manifest.json').read_text())['status'] == 'failed'

        monkeypatch.setattr(target=stability, name='record_responses', value=original)
        monkeypatch.setattr(
            target=stability.EpisodeBrain,
            name='features',
            value=lambda self, kinds: np.full(
                shape=(self.batch_size, len(self.readout_idx) * len(kinds)),
                fill_value=np.nan,
                dtype=np.float32,
            ),
        )
        with pytest.raises(expected_exception=ValueError, match='must be finite'):
            run_stability(
                cfg=cfg,
                paths=paths,
                output=paths.runs / 'nonfinite',
                sample_ids=ids,
                membership_sha256=sha256_obj(obj=ids),
            )

"""Verify deterministic observation order, overlap bounds, crops, and rendering."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets import registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.preprocess import prepare_dataset
from flystate.datasets.subset import Sample
from flystate.episodes.episode import EpisodeBuilder
from flystate.episodes.trajectory import TrajectoryError, make_boxes
from flystate.experiments.config import (
    EpisodesConfig,
    ExperimentConfig,
    TrajectoryConfig,
    effective_yaml,
)
from flystate.settings import get_paths


class TestTrajectories:
    """Check repeatability, grid conventions, and pairwise shared-area limits."""

    def test_random_geometry(self) -> None:
        """Verify all pairs across 500 seeded eight-observation trajectories."""
        first = EpisodeBuilder(episodes=EpisodesConfig(), image_size=128)
        second = EpisodeBuilder(episodes=EpisodesConfig(), image_size=128)
        np.testing.assert_array_equal(
            actual=first.boxes_for(sample_id='s1'), desired=second.boxes_for(sample_id='s1')
        )
        assert not np.array_equal(
            a1=first.boxes_for(sample_id='s1'), a2=first.boxes_for(sample_id='s2')
        )
        for index in range(500):
            boxes = first.boxes_for(sample_id=f'sample-{index}')
            assert boxes.dtype == np.int32 and boxes.shape == (8, 4)
            assert boxes.min() >= 0 and boxes.max() <= 128
            for left, right in combinations(boxes, 2):
                width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
                height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
                assert width * height <= 0.25 * 32 * 32
        assert first.trajectory_hash(sample_ids=['s1', 's2']) == second.trajectory_hash(
            sample_ids=['s1', 's2']
        )
        assert first.trajectory_hash(sample_ids=['s1', 's2']) != first.trajectory_hash(
            sample_ids=['s2', 's1']
        )

    @pytest.mark.parametrize('steps,offsets', [(4, [0, 96]), (16, [0, 32, 64, 96])])
    def test_grid_and_permutation(self, steps: int, offsets: list[int]) -> None:
        """Match exact row-major offsets and preserve their set under permutation.

        :param steps: Square observation count.
        :type steps: int
        :param offsets: Expected x/y pixel offsets.
        :type offsets: list[int]
        """
        raster = make_boxes(
            sample_id='s1',
            image_size=128,
            window=32,
            steps=steps,
            trajectory=TrajectoryConfig(strategy='raster'),
        )
        expected = [[x, y, x + 32, y + 32] for y in offsets for x in offsets]
        np.testing.assert_array_equal(actual=raster, desired=expected)
        permuted = make_boxes(
            sample_id='s1',
            image_size=128,
            window=32,
            steps=steps,
            trajectory=TrajectoryConfig(strategy='permuted'),
        )
        assert sorted(permuted.tolist()) == sorted(raster.tolist())
        assert not np.array_equal(a1=permuted, a2=raster)

    def test_impossible_geometry(self) -> None:
        """Fail bounded rejection sampling instead of violating the requested overlap."""
        with pytest.raises(expected_exception=TrajectoryError, match='1000 draws'):
            make_boxes(
                sample_id='impossible',
                image_size=128,
                window=32,
                steps=40,
                trajectory=TrajectoryConfig(max_overlap=0),
            )
        with pytest.raises(expected_exception=TrajectoryError, match='positive steps'):
            make_boxes(
                sample_id='s', image_size=16, window=32, steps=1, trajectory=TrajectoryConfig()
            )
        with pytest.raises(expected_exception=TrajectoryError, match='square step count'):
            make_boxes(
                sample_id='s',
                image_size=128,
                window=32,
                steps=8,
                trajectory=TrajectoryConfig(strategy='raster'),
            )
        with pytest.raises(expected_exception=TrajectoryError, match='fit inside'):
            EpisodeBuilder(episodes=EpisodesConfig(), image_size=16)


class TestEpisodes:
    """Preserve exact observation pixels and render metadata through the CLI."""

    def test_crops_and_positions(self) -> None:
        """Compare every crop and normalized center to independently indexed source pixels."""
        image = np.random.default_rng(seed=3).integers(
            low=0, high=256, size=(128, 128, 3), dtype=np.uint8
        )
        sample = Sample(sample_id='s1', filename='000001.jpg', identity=1, label=0, split='train')
        builder = EpisodeBuilder(episodes=EpisodesConfig(), image_size=128)
        episode = builder.build(sample=sample, image=image)
        assert episode.target == 0 and episode.split == 'train'
        assert episode.observations.shape == (8, 32, 32, 3)
        assert episode.positions.dtype == np.float32
        assert np.all(episode.positions >= 0) and np.all(episode.positions <= 1)
        for index, (x0, y0, x1, y1) in enumerate(episode.boxes):
            np.testing.assert_array_equal(
                actual=episode.observations[index], desired=image[y0:y1, x0:x1]
            )
            np.testing.assert_allclose(
                actual=episode.positions[index], desired=[(x0 + 16) / 128, (y0 + 16) / 128]
            )
        episode.observations.fill(0)
        assert image.any()
        with pytest.raises(expected_exception=ValueError, match='RGB uint8'):
            builder.build(sample=sample, image=image[:32])

    def test_cli(self, synthetic_celeba_dir: Path, tmp_path: Path) -> None:
        """Render a complete contact sheet and report configuration or sample errors.

        :param synthetic_celeba_dir: Generated offline faces.
        :type synthetic_celeba_dir: Path
        :param tmp_path: Config directory.
        :type tmp_path: Path
        """
        paths = get_paths()
        registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
        registry.set_validation(
            paths=paths,
            name='celeba',
            report=CelebAAdapter(root=synthetic_celeba_dir, expected=None).validate(),
            full=False,
        )
        cfg = ExperimentConfig.model_validate(
            obj={'name': 'fixture', 'dataset': {'subset': {'n_identities': 4}}}
        )
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        source = tmp_path / 'experiment.yaml'
        source.write_text(data=effective_yaml(cfg=cfg))
        runner = CliRunner()
        args = [
            'episode',
            'show',
            str(source),
            prepared.samples[0].sample_id,
            '--out',
            'episode.png',
        ]
        result = runner.invoke(app=app, args=[*args, '--json'])
        assert result.exit_code == 0, result.exception
        payload = json.loads(s=result.stdout)
        assert len(payload['boxes']) == len(payload['positions']) == 8
        with Image.open(fp=paths.home / 'episode.png') as image:
            assert image.size == (1152, 384)
        assert runner.invoke(app=app, args=['episode', 'show', '--help']).exit_code == 0
        for json_flag in ([], ['--json']):
            missing = runner.invoke(
                app=app,
                args=['episode', 'show', str(source), 'unknown', '--out', 'bad.png', *json_flag],
            )
            assert missing.exit_code == 1
            assert 'not in the configured subset' in missing.stderr
        invalid = runner.invoke(
            app=app, args=['episode', 'show', 'absent.yaml', 's1', '--out', 'bad.png', '--json']
        )
        assert invalid.exit_code == 2

"""Test image-content cache identity, train-only fitting, and atomic preparation."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets import preprocess, registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.errors import DatasetError
from flystate.datasets.preprocess import PreparedDataset, prepare_dataset
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_file
from flystate.settings import get_paths
from flystate.storage.json import write_json


@pytest.fixture
def prepared_source(synthetic_celeba_dir: Path) -> ExperimentConfig:
    """Register a validated synthetic dataset and a feasible four-class experiment.

    :param synthetic_celeba_dir: Eight-identity source dataset.
    :type synthetic_celeba_dir: Path
    :returns: Small experiment using twenty images per identity.
    :rtype: ExperimentConfig
    """
    paths = get_paths()
    registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
    registry.set_validation(
        paths=paths,
        name='celeba',
        report=CelebAAdapter(root=synthetic_celeba_dir, expected=None).validate(full=True),
        full=True,
    )
    return ExperimentConfig.model_validate(
        obj={'name': 'fixture', 'dataset': {'subset': {'n_identities': 4}}}
    )


def rewrite_digest(prepared: PreparedDataset, name: str) -> None:
    """Update one test cache digest to exercise structural validation after integrity succeeds.

    :param prepared: Test-owned cache.
    :type prepared: PreparedDataset
    :param name: Modified artifact basename.
    :type name: str
    """
    path = prepared.directory / 'meta.json'
    meta = json.loads(s=path.read_text())
    meta['artifacts'][name] = sha256_file(path=prepared.directory / name)
    write_json(path=path, value=meta)


class TestPreprocessing:
    """Verify stable artifacts, source-sensitive identities and leakage-free templates."""

    def test_reuse_and_source_pixels(
        self, prepared_source: ExperimentConfig, synthetic_celeba_dir: Path
    ) -> None:
        """Reuse unchanged artifacts and invalidate the key when selected JPEG pixels change.

        :param prepared_source: Validated fixture experiment.
        :type prepared_source: ExperimentConfig
        :param synthetic_celeba_dir: Source JPEG root.
        :type synthetic_celeba_dir: Path
        """
        paths = get_paths()
        first = prepare_dataset(cfg=prepared_source, paths=paths)
        timestamps = {path.name: path.stat().st_mtime_ns for path in first.directory.iterdir()}
        second = prepare_dataset(cfg=prepared_source, paths=paths)
        assert first.key == second.key and first.fingerprint == second.fingerprint
        assert timestamps == {
            path.name: path.stat().st_mtime_ns for path in first.directory.iterdir()
        }
        assert isinstance(first.images, np.memmap) and not first.images.flags.writeable
        assert first.images.shape == (80, 128, 128, 3)
        assert not first.template.flags.writeable
        source = CelebAAdapter(root=synthetic_celeba_dir, expected=None).locate().images_dir
        Image.new(mode='RGB', size=(178, 218), color='white').save(
            fp=source / first.samples[0].filename
        )
        changed = prepare_dataset(cfg=prepared_source, paths=paths)
        assert changed.key != first.key and changed.fingerprint != first.fingerprint
        assert first.directory.exists()
        altered = prepared_source.model_dump(mode='json')
        altered['dataset']['preprocess']['interocular'] = 42
        geometry = prepare_dataset(cfg=ExperimentConfig.model_validate(obj=altered), paths=paths)
        assert geometry.key != changed.key

    def test_training_only_template(
        self, prepared_source: ExperimentConfig, synthetic_celeba_dir: Path
    ) -> None:
        """Changing only held-out landmarks cannot change the fitted alignment template.

        :param prepared_source: Validated fixture experiment.
        :type prepared_source: ExperimentConfig
        :param synthetic_celeba_dir: Original annotation root.
        :type synthetic_celeba_dir: Path
        """
        paths = get_paths()
        first = prepare_dataset(cfg=prepared_source, paths=paths)
        held_out = next(item.filename for item in first.samples if item.split == 'val')
        adapter = CelebAAdapter(root=synthetic_celeba_dir, expected=None)
        path = adapter.locate().landmarks_file
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines[2:], start=2):
            tokens = line.split()
            if tokens[0] == held_out:
                lines[index] = (
                    tokens[0] + ' ' + ' '.join(str(float(value) + 10) for value in tokens[1:])
                )
                break
        path.write_text(data='\n'.join(lines) + '\n')
        registry.set_validation(paths=paths, name='celeba', report=adapter.validate(), full=False)
        changed = prepare_dataset(cfg=prepared_source, paths=paths)
        assert changed.key != first.key
        np.testing.assert_array_equal(actual=changed.template, desired=first.template)

    @pytest.mark.parametrize('corruption', ['digest', 'identity', 'index', 'image', 'template'])
    def test_corrupt_cache(self, prepared_source: ExperimentConfig, corruption: str) -> None:
        """Reject corrupted artifacts and structurally inconsistent metadata without overwriting.

        :param prepared_source: Validated fixture experiment.
        :type prepared_source: ExperimentConfig
        :param corruption: Artifact corruption case.
        :type corruption: str
        """
        prepared = prepare_dataset(cfg=prepared_source, paths=get_paths())
        if corruption == 'identity':
            path = prepared.directory / 'meta.json'
            meta = json.loads(s=path.read_text())
            meta['key'] = 'wrong'
            write_json(path=path, value=meta)
        elif corruption in {'digest', 'template'}:
            write_json(path=prepared.directory / 'template.json', value=[])
            if corruption == 'template':
                rewrite_digest(prepared=prepared, name='template.json')
        elif corruption == 'image':
            np.save(
                file=prepared.directory / 'images.npy', arr=np.zeros(shape=(1, 2), dtype=np.uint8)
            )
            rewrite_digest(prepared=prepared, name='images.npy')
        else:
            rows = [
                {'row': index, **asdict(obj=sample)}
                for index, sample in enumerate(prepared.samples)
            ]
            rows[0]['label'] = 999
            pq.write_table(
                table=pa.Table.from_pylist(mapping=rows), where=prepared.directory / 'index.parquet'
            )
            rewrite_digest(prepared=prepared, name='index.parquet')
        with pytest.raises(expected_exception=DatasetError, match='Invalid preprocessing cache'):
            prepare_dataset(cfg=prepared_source, paths=get_paths())

    def test_interrupted_build(
        self, prepared_source: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clean only owned temporary artifacts and reject source changes mid-build.

        :param prepared_source: Validated fixture experiment.
        :type prepared_source: ExperimentConfig
        :param monkeypatch: Simulated decode and source-fingerprint changes.
        :type monkeypatch: pytest.MonkeyPatch
        """
        paths = get_paths()
        with monkeypatch.context() as patch:
            patch.setattr(
                target=preprocess, name='align_face', value=Mock(side_effect=KeyboardInterrupt())
            )
            with pytest.raises(expected_exception=KeyboardInterrupt):
                prepare_dataset(cfg=prepared_source, paths=paths)
        assert not list(paths.preprocess.iterdir())
        with monkeypatch.context() as patch:
            patch.setattr(
                target=preprocess,
                name='selected_images_hash',
                value=Mock(side_effect=['before', 'after']),
            )
            with pytest.raises(expected_exception=DatasetError, match='changed during'):
                prepare_dataset(cfg=prepared_source, paths=paths)
        assert not list(paths.preprocess.iterdir())
        ready = prepare_dataset(cfg=prepared_source, paths=paths)
        temporary = ready.directory.with_name(ready.key + '.tmp')
        ready.directory.rename(target=temporary)
        with pytest.raises(expected_exception=DatasetError, match='already exists'):
            prepare_dataset(cfg=prepared_source, paths=paths)
        assert temporary.exists()

    def test_cli(self, prepared_source: ExperimentConfig, tmp_path: Path) -> None:
        """Prepare a cache and inspect an aligned sample through the command interface.

        :param prepared_source: Validated fixture experiment.
        :type prepared_source: ExperimentConfig
        :param tmp_path: Experiment YAML directory.
        :type tmp_path: Path
        """
        source = tmp_path / 'experiment.yaml'
        source.write_text(data=effective_yaml(cfg=prepared_source))
        runner = CliRunner()
        result = runner.invoke(app=app, args=['dataset', 'prepare', str(source), '--json'])
        assert result.exit_code == 0, result.exception
        assert json.loads(s=result.stdout)['counts'] == {'train': 56, 'val': 12, 'test': 12}
        result = runner.invoke(
            app=app,
            args=[
                'dataset',
                'inspect',
                'celeba',
                '000001.jpg',
                '--config',
                str(source),
                '--out',
                'aligned.png',
                '--json',
            ],
        )
        assert result.exit_code == 0, result.exception
        with Image.open(fp=get_paths().home / 'aligned.png') as image:
            assert image.size == (128, 128)
        for command in (['prepare'], ['inspect', 'celeba', '000001.jpg', '--config']):
            result = runner.invoke(
                app=app, args=['dataset', *command, str(tmp_path / 'absent'), '--json']
            )
            assert result.exit_code == 2
        get_paths().datasets.joinpath('registry.json').unlink()
        result = runner.invoke(app=app, args=['dataset', 'prepare', str(source), '--json'])
        assert result.exit_code == 1
        assert runner.invoke(app=app, args=['dataset', 'prepare', '--help']).exit_code == 0

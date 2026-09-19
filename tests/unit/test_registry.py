"""Test durable registration and the complete local dataset CLI lifecycle."""

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.datasets import registry
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.errors import DatasetError
from flystate.settings import get_paths


class TestRegistry:
    """Preserve explicit registration and invalidate stale validation evidence."""

    def test_validation_lifecycle(self, synthetic_celeba_dir: Path) -> None:
        """Require validation, detect source mutation, and clear evidence on registration.

        :param synthetic_celeba_dir: Synthetic official-layout source.
        :type synthetic_celeba_dir: Path
        """
        paths = get_paths()
        assert registry.entries(paths=paths) == {}
        with pytest.raises(expected_exception=DatasetError, match='dataset validate celeba'):
            registry.require_validated(paths=paths, name='celeba')
        entry = registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
        assert entry['path'] == str(synthetic_celeba_dir.resolve())
        with pytest.raises(expected_exception=DatasetError, match='not passed validation'):
            registry.require_validated(paths=paths, name='celeba')
        adapter = CelebAAdapter(root=synthetic_celeba_dir, expected=None)
        registry.set_validation(
            paths=paths, name='celeba', report=adapter.validate(full=True), full=True
        )
        assert registry.require_validated(paths=paths, name='celeba') == synthetic_celeba_dir
        source = adapter.locate().identity_file
        source.write_text(data=source.read_text() + '\n')
        with pytest.raises(expected_exception=DatasetError, match='changed after validation'):
            registry.require_validated(paths=paths, name='celeba')
        registry.register(paths=paths, name='celeba', root=synthetic_celeba_dir)
        assert registry.get(paths=paths, name='celeba')['validation'] is None

    def test_bad_registry(self, tmp_path: Path) -> None:
        """Reject invalid names, missing roots, and corrupt registry objects.

        :param tmp_path: Test-owned root.
        :type tmp_path: Path
        """
        paths = get_paths()
        with pytest.raises(expected_exception=DatasetError, match='Only celeba'):
            registry.register(paths=paths, name='lfw', root=tmp_path)
        with pytest.raises(expected_exception=DatasetError, match='not a directory'):
            registry.register(paths=paths, name='celeba', root=tmp_path / 'absent')
        paths.datasets.mkdir(parents=True)
        for contents in ('[]', '{"celeba":1}', '{'):
            (paths.datasets / 'registry.json').write_text(data=contents)
            with pytest.raises(expected_exception=DatasetError, match='Cannot read'):
                registry.entries(paths=paths)

    def test_cli_lifecycle(self, synthetic_celeba_dir: Path) -> None:
        """Register, validate, describe, list, and inspect without real images.

        :param synthetic_celeba_dir: Synthetic official-layout source.
        :type synthetic_celeba_dir: Path
        """
        runner = CliRunner()
        commands = [
            ['register', 'celeba', str(synthetic_celeba_dir)],
            ['validate', 'celeba', '--full', '--expected-counts', 'none'],
            ['info', 'celeba'],
            ['list'],
            ['inspect', 'celeba', 'celeba-000001', '--out', 'preview.png'],
            ['inspect', 'celeba', '000002.jpg'],
        ]
        for command in commands:
            result = runner.invoke(app=app, args=['dataset', *command, '--json'])
            assert result.exit_code == 0, result.exception
            assert isinstance(json.loads(s=result.stdout), dict)
        with Image.open(fp=get_paths().home / 'preview.png') as image:
            assert image.size == (178, 218)
            assert image.format == 'PNG'
        info = json.loads(
            s=runner.invoke(app=app, args=['dataset', 'info', 'celeba', '--json']).stdout
        )
        assert info['images_per_identity'] == {
            'min': 25,
            'median': 25.0,
            'max': 25,
            'at_least_20': 8,
        }
        assert 'non-commercial research' in info['license']
        for command_name in ('register', 'validate', 'info', 'list', 'inspect'):
            assert runner.invoke(app=app, args=['dataset', command_name, '--help']).exit_code == 0
        assert runner.invoke(app=app, args=['dataset', 'info', 'celeba']).exit_code == 0
        assert (
            runner.invoke(app=app, args=['dataset', 'validate', 'celeba', '--json']).exit_code == 1
        )
        assert (
            runner.invoke(
                app=app, args=['dataset', 'validate', 'celeba', '--expected-counts', 'bad']
            ).exit_code
            == 2
        )
        assert runner.invoke(app=app, args=['dataset', 'info', 'lfw']).exit_code == 2
        for sample in ('celeba-999999', '../bad.jpg'):
            result = runner.invoke(app=app, args=['dataset', 'inspect', 'celeba', sample, '--json'])
            assert result.exit_code == 1
            assert 'error' in json.loads(s=result.stdout)
        result = runner.invoke(
            app=app, args=['dataset', 'inspect', 'celeba', '000001.jpg', '--out', '../escape.png']
        )
        assert result.exit_code == 1
        assert result.stdout == ''

    def test_cli_failures(self, tmp_path: Path) -> None:
        """Report missing registrations and corrupt registry files consistently.

        :param tmp_path: Empty dataset parent.
        :type tmp_path: Path
        """
        runner = CliRunner()
        for command in (
            ['register', 'celeba', str(tmp_path / 'absent')],
            ['validate', 'celeba'],
            ['info', 'celeba'],
            ['inspect', 'celeba', '000001.jpg'],
        ):
            result = runner.invoke(app=app, args=['dataset', *command, '--json'])
            assert result.exit_code == 1
            assert 'error' in json.loads(s=result.stdout)
        paths = get_paths()
        paths.datasets.mkdir(parents=True)
        (paths.datasets / 'registry.json').write_text(data='bad')
        assert runner.invoke(app=app, args=['dataset', 'list', '--json']).exit_code == 1

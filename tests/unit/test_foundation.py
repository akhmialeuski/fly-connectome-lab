"""Verify storage boundaries, content identities, logging, and the public CLI."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flystate import __version__
from flystate.cli.main import app
from flystate.hashing import canonical_json, sha256_file, sha256_obj, sha256_text, stable_int
from flystate.log import configure_logging, get_logger
from flystate.settings import ensure_dirs, get_paths, output_path


class TestFoundation:
    """Exercise the installation contract without datasets or network access."""

    def test_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve defaults, create directories, and reject escaping exports.

        :param tmp_path: Temporary working directory.
        :type tmp_path: Path
        :param monkeypatch: Isolated environment changes.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.delenv(name='FLYSTATE_HOME')
        monkeypatch.chdir(path=tmp_path)
        paths = get_paths()
        assert paths.home == tmp_path
        ensure_dirs(paths=paths)
        assert all(value.is_dir() for value in asdict(obj=paths).values())
        monkeypatch.setenv(name='FLYSTATE_HOME', value=str(tmp_path / 'other'))
        assert get_paths().home == tmp_path / 'other'
        assert get_paths(home=tmp_path).home == tmp_path
        assert output_path(path=Path('preview.png'), paths=paths) == tmp_path / 'preview.png'
        for invalid in ('../escape.png', 'src/generated.png'):
            with pytest.raises(expected_exception=ValueError):
                output_path(path=Path(invalid), paths=paths)

    def test_hashes(self, tmp_path: Path) -> None:
        """Compare portable hashes to independent hashlib calculations.

        :param tmp_path: Temporary file directory.
        :type tmp_path: Path
        """
        source = tmp_path / 'content.txt'
        source.write_text(data='abc', encoding='utf-8')
        expected = hashlib.sha256(string=b'abc')
        assert sha256_file(path=source, chunk_size=1) == expected.hexdigest()
        assert sha256_text(text='abc') == expected.hexdigest()
        assert stable_int(key='abc') == int.from_bytes(expected.digest()[:8], byteorder='big')
        assert sha256_obj(obj={'b': 2, 'a': 1}) == sha256_obj(obj={'a': 1, 'b': 2})
        assert canonical_json(obj={'b': 2, 'a': 1}) == '{"a":1,"b":2}'
        with pytest.raises(expected_exception=ValueError):
            sha256_file(path=source, chunk_size=0)

    def test_logging(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Replace handlers twice, emit once, and check both destinations.

        :param tmp_path: Log directory.
        :type tmp_path: Path
        :param capsys: Captured console streams.
        :type capsys: pytest.CaptureFixture[str]
        """
        destination = tmp_path / 'events.jsonl'
        configure_logging(verbosity=0, json_path=destination)
        configure_logging(verbosity=1, json_path=destination)
        get_logger(name='test').bind(run_id='test-run').info('written')
        lines = destination.read_text(encoding='utf-8').splitlines()
        assert len(lines) == 1
        event = json.loads(s=lines[0])
        assert {'timestamp', 'event', 'level', 'run_id'} <= event.keys()
        assert 'written' in capsys.readouterr().err
        configure_logging(verbosity=-1)

    def test_cli(self) -> None:
        """Invoke help, version, and both output modes through the public CLI."""
        runner = CliRunner()
        result = runner.invoke(app=app, args=['--help'])
        assert result.exit_code == 0, result.output
        for name in ('brain', 'dataset', 'episode', 'experiment', 'trace'):
            assert name in result.output
        assert __version__ in runner.invoke(app=app, args=['--version']).output
        result = runner.invoke(app=app, args=['paths', '--json'])
        assert result.exit_code == 0, result.exception
        assert len(json.loads(s=result.stdout)) == 8
        assert 'home:' in runner.invoke(app=app, args=['paths']).stdout

"""Verify unique run allocation, atomic artifacts, lifecycle guards, and source provenance."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from flystate.experiments.config import ExperimentConfig
from flystate.readouts.linear import save_readouts
from flystate.settings import get_paths
from flystate.storage.environment import environment_report, git_state
from flystate.storage.runs import (
    RunError,
    create_run,
    load_manifest,
    resolve_run,
    update_manifest,
    write_run_json,
    write_run_parquet,
    write_run_text,
)


class TestRuns:
    """Exercise immutable completion and safe artifact paths without fitting a model."""

    def test_lifecycle(self) -> None:
        """Reserve colliding IDs, write artifacts, complete a run, and reject every write helper."""
        cfg = ExperimentConfig(name='run-test')
        paths = get_paths()
        with patch(target='flystate.storage.runs.datetime') as clock:
            clock.now.return_value = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
            first = create_run(
                paths=paths, cfg=cfg, original_yaml='name: run-test\n', extra_manifest={}
            )
            second = create_run(
                paths=paths, cfg=cfg, original_yaml='name: run-test\n', extra_manifest={}
            )
        assert re.fullmatch(pattern=r'\d{8}-\d{6}-[a-z0-9-]+-[0-9a-f]{6}', string=first.name)
        assert second.name == f'{first.name}-2'
        assert resolve_run(paths=paths, run=first.name) == first
        assert resolve_run(paths=paths, run=str(first)) == first
        assert (first / 'config.original.yaml').read_text() == 'name: run-test\n'
        write_run_parquet(run_dir=first, relative='metrics/check.parquet', rows=[{'value': 1}])
        assert pq.read_table(source=first / 'metrics/check.parquet').to_pylist() == [{'value': 1}]
        for relative in ('../outside.json', str(paths.home / 'outside.json')):
            with pytest.raises(expected_exception=RunError, match='within'):
                write_run_json(run_dir=first, relative=relative, value={})
        with pytest.raises(expected_exception=RunError, match='identity'):
            update_manifest(run_dir=first, run_id='changed')
        with pytest.raises(expected_exception=RunError, match='status'):
            update_manifest(run_dir=first, status='unknown')
        update_manifest(run_dir=first, status='completed', finished_utc='2026-01-02T03:04:06+00:00')
        assert load_manifest(run_dir=first)['status'] == 'completed'
        with pytest.raises(expected_exception=RunError, match='immutable'):
            update_manifest(run_dir=first, status='failed')
        with pytest.raises(expected_exception=RunError):
            write_run_json(run_dir=first, relative='summary.json', value={})
        with pytest.raises(expected_exception=RunError):
            write_run_text(run_dir=first, relative='text.txt', value='changed')
        with pytest.raises(expected_exception=RunError):
            write_run_parquet(run_dir=first, relative='metrics/new.parquet', rows=[])
        with pytest.raises(expected_exception=RunError):
            save_readouts(path=first / 'model', readouts=[], meta={})

    def test_failures(self) -> None:
        """Persist failed creation and clean temporary artifacts after later write failures."""
        paths = get_paths()
        with (
            patch(
                target='flystate.storage.runs.environment_report',
                side_effect=RuntimeError('environment failed'),
            ),
            pytest.raises(expected_exception=RuntimeError),
        ):
            create_run(
                paths=paths,
                cfg=ExperimentConfig(name='failed'),
                original_yaml='',
                extra_manifest={},
            )
        failed = next(paths.runs.iterdir())
        assert load_manifest(run_dir=failed)['status'] == 'failed'
        before = set(failed.iterdir())
        with (
            patch(target='flystate.storage.runs.os.fsync', side_effect=OSError('disk')),
            pytest.raises(expected_exception=OSError),
        ):
            write_run_text(run_dir=failed, relative='new.txt', value='new')
        assert set(failed.iterdir()) == before
        with (
            patch(target='flystate.storage.runs.pq.write_table', side_effect=OSError('disk')),
            pytest.raises(expected_exception=OSError),
        ):
            write_run_parquet(run_dir=failed, relative='new.parquet', rows=[{'x': 1}])
        assert set(failed.iterdir()) == before
        meta = load_manifest(run_dir=failed)
        with patch(
            target='flystate.storage.runs.tempfile.NamedTemporaryFile',
            side_effect=OSError('disk full'),
        ):
            with pytest.raises(expected_exception=OSError, match='disk full'):
                write_run_text(run_dir=failed, relative='new.txt', value='new')
            with pytest.raises(expected_exception=OSError, match='disk full'):
                write_run_parquet(run_dir=failed, relative='new.parquet', rows=[{'x': 1}])
        assert set(failed.iterdir()) == before
        meta['run_id'] = 'wrong'
        (failed / 'manifest.json').write_text(data=json.dumps(obj=meta))
        with pytest.raises(expected_exception=RunError):
            load_manifest(run_dir=failed)
        with pytest.raises(expected_exception=RunError):
            load_manifest(run_dir=paths.runs / 'absent')

    def test_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Record current numerical versions and distinguish a checkout from an unrelated directory.

        :param tmp_path: Directory outside Git.
        :type tmp_path: Path
        :param monkeypatch: Scoped working-directory change.
        :type monkeypatch: pytest.MonkeyPatch
        """
        report = environment_report()
        assert report['python'].startswith('3.12.') and report['numba_threads'] > 0
        assert set(report['versions']) == {
            'numpy',
            'scipy',
            'numba',
            'scikit-learn',
            'zarr',
            'pyarrow',
        }
        assert git_state()['git_commit'] is not None
        monkeypatch.chdir(path=tmp_path)
        assert git_state() == {'git_commit': None, 'git_dirty': None}

"""Exercise measured throughput and durable reports on an offline connectome."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numba
import numpy as np
import pytest
from typer.testing import CliRunner

from flystate.brain import benchmark
from flystate.cli.main import app
from flystate.settings import get_paths
from flystate.storage import json as json_storage
from flystate.storage.json import write_json


class TestBenchmark:
    """Validate benchmark arithmetic, persistence, resource restoration, and CLI."""

    def test_grid_and_cli(self, synthetic_brain_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run a real four-cell synthetic grid and a two-second sustained measurement.

        :param synthetic_brain_dir: Offline connectome files.
        :type synthetic_brain_dir: Path
        :param monkeypatch: Frozen report timestamp for collision testing.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=benchmark,
            name='datetime',
            value=Mock(now=Mock(return_value=datetime(year=2026, month=1, day=1, tzinfo=UTC))),
        )
        runner = CliRunner()
        args = [
            'brain',
            'benchmark',
            '--brain-dir',
            str(synthetic_brain_dir),
            '--threads',
            '1,2',
            '--batch',
            '1,2',
            '--warmup',
            '5',
            '--steps',
            '20',
            '--sustained-seconds',
            '2',
            '--input-neurons',
            '30',
            '--json',
        ]
        original_threads = numba.get_num_threads()
        result = runner.invoke(app=app, args=args)
        assert result.exit_code == 0, result.exception
        assert numba.get_num_threads() == original_threads
        report = json.loads(s=result.stdout)
        assert set(report) >= {
            'schema_version',
            'created_utc',
            'host',
            'versions',
            'brain',
            'load',
            'grid',
            'best',
            'sustained',
            'episode_estimates',
            'trace_build_estimates',
        }
        assert len(report['grid']) == 4
        assert report['best']['ms_per_episode_step'] == min(
            row['ms_per_episode_step'] for row in report['grid']
        )
        assert report['sustained']['elapsed_seconds'] >= 2
        assert report['sustained']['throttle_ratio'] == pytest.approx(expected=1.0)
        saved = Path(report['report_path'])
        assert json.loads(s=saved.read_text()) == report
        latest = get_paths().runs / 'benchmarks' / 'latest.json'
        assert json.loads(s=latest.read_text()) == report
        second_output = runner.invoke(
            app=app,
            args=[
                'brain',
                'benchmark',
                '--brain-dir',
                str(synthetic_brain_dir),
                '--threads',
                '1',
                '--batch',
                '1',
                '--warmup',
                '0',
                '--steps',
                '2',
                '--sustained-seconds',
                '0.01',
                '--input-neurons',
                '1',
            ],
        )
        assert second_output.exit_code == 0, second_output.exception
        assert 'ms/episode-step' in second_output.stdout
        second = json.loads(s=latest.read_text())
        assert second['report_path'] != str(saved)
        assert json.loads(s=saved.read_text()) == report
        assert benchmark.resolve_threads(paths=get_paths(), configured=None) == 1
        assert runner.invoke(app=app, args=['brain', 'benchmark', '--help']).exit_code == 0

    def test_estimates_and_windows(self) -> None:
        """Check independent units and proportional treatment of boundary blocks."""
        report = benchmark.cost_estimates(ms_per_episode_step=10)
        episodes = report['episode_estimates']
        builds = report['trace_build_estimates']
        assert isinstance(episodes, list) and isinstance(builds, list)
        assert episodes[1]['seconds_per_episode'] == 0.4
        assert builds[4]['minutes'] == pytest.approx(expected=13.333333333)
        assert benchmark._window_time(
            samples=[(0, 10), (10, 30)], start=5, end=20, batch_size=2
        ) == pytest.approx(expected=750)
        with pytest.raises(expected_exception=ValueError):
            benchmark.cost_estimates(ms_per_episode_step=float('nan'))

    def test_thread_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Respect explicit settings and reject invalid persisted thread values.

        :param monkeypatch: Fixed fallback CPU count.
        :type monkeypatch: pytest.MonkeyPatch
        """
        paths = get_paths()
        monkeypatch.setattr(target=benchmark.os, name='cpu_count', value=Mock(return_value=3))
        assert benchmark.default_threads() == (1, 2, 3)
        assert benchmark.resolve_threads(paths=paths, configured=2) == 2
        assert benchmark.resolve_threads(paths=paths, configured=None) == 3
        destination = paths.runs / 'benchmarks' / 'latest.json'
        write_json(path=destination, value={'best': {'threads': 1}})
        assert benchmark.resolve_threads(paths=paths, configured=None) == 1
        for invalid in (0, True, '2'):
            write_json(path=destination, value={'best': {'threads': invalid}})
            with pytest.raises(expected_exception=ValueError, match='positive integer'):
                benchmark.resolve_threads(paths=paths, configured=None)
        destination.write_text(data='broken JSON')
        with pytest.raises(expected_exception=ValueError, match='Invalid saved benchmark'):
            benchmark.resolve_threads(paths=paths, configured=None)
        with pytest.raises(expected_exception=ValueError, match='positive integer'):
            benchmark.resolve_threads(paths=paths, configured=0)

    def test_invalid_cli(self) -> None:
        """Reject malformed grids and missing connectomes with useful exit codes."""
        runner = CliRunner()
        for value in ('', 'zero', '0', '1,'):
            result = runner.invoke(app=app, args=['brain', 'benchmark', '--threads', value])
            assert result.exit_code == 2
        result = runner.invoke(app=app, args=['brain', 'benchmark', '--json'])
        assert result.exit_code == 1
        assert 'brain download' in json.loads(s=result.stdout)['error']

    def test_invalid_measurements(self, synthetic_brain_dir: Path) -> None:
        """Reject invalid grids, durations, and input population sizes before timing.

        :param synthetic_brain_dir: Offline connectome files.
        :type synthetic_brain_dir: Path
        """
        with pytest.raises(expected_exception=ValueError, match='positive'):
            benchmark.run_benchmark(
                paths=get_paths(), brain_dir=synthetic_brain_dir, threads=[], batches=[1]
            )
        with pytest.raises(expected_exception=ValueError, match='duration'):
            benchmark.run_benchmark(
                paths=get_paths(),
                brain_dir=synthetic_brain_dir,
                threads=[1],
                batches=[1],
                sustained_seconds=0,
            )
        previous = numba.get_num_threads()
        with pytest.raises(expected_exception=ValueError, match='population has'):
            benchmark.run_benchmark(
                paths=get_paths(), brain_dir=synthetic_brain_dir, threads=[1], batches=[1]
            )
        assert numba.get_num_threads() == previous

    def test_optional_frequency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Treat absent CPU telemetry as unavailable rather than a failed experiment.

        :param monkeypatch: Isolated host telemetry responses.
        :type monkeypatch: pytest.MonkeyPatch
        """
        for response in (None, SimpleNamespace(current=2000.0)):
            monkeypatch.setattr(
                target=benchmark.psutil, name='cpu_freq', value=Mock(return_value=response)
            )
            assert benchmark._cpu_frequency() == (None if response is None else 2000.0)
        monkeypatch.setattr(
            target=benchmark.psutil, name='cpu_freq', value=Mock(side_effect=OSError('absent'))
        )
        assert benchmark._cpu_frequency() is None

    def test_host_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read model names when available and retain the platform fallback otherwise.

        :param monkeypatch: Simulated procfs and processor string.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=benchmark.platform, name='processor', value=Mock(return_value='fallback')
        )
        for exists, content, expected in (
            (False, '', 'fallback'),
            (True, 'processor: 0', 'fallback'),
            (True, 'model name: Test CPU', 'Test CPU'),
        ):
            source = Mock(is_file=Mock(return_value=exists), read_text=Mock(return_value=content))
            monkeypatch.setattr(target=benchmark, name='Path', value=Mock(return_value=source))
            assert benchmark.host_info()['cpu_model'] == expected


class TestAtomicJson:
    """Protect existing metadata from partial or invalid writes."""

    def test_replacement_and_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Publish one object and preserve it if the subsequent rename fails.

        :param tmp_path: Output directory.
        :type tmp_path: Path
        :param monkeypatch: Atomic rename failure.
        :type monkeypatch: pytest.MonkeyPatch
        """
        destination = tmp_path / 'report.json'
        write_json(path=destination, value={'complete': True})
        monkeypatch.setattr(
            target=Path, name='replace', value=Mock(side_effect=OSError('rename denied'))
        )
        with pytest.raises(expected_exception=OSError):
            write_json(path=destination, value={'complete': False})
        assert json.loads(s=destination.read_text()) == {'complete': True}
        assert list(tmp_path.iterdir()) == [destination]

    def test_temporary_creation_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Propagate temporary-file creation failure without publishing a destination.

        :param tmp_path: Output directory.
        :type tmp_path: Path
        :param monkeypatch: File creation failure.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=json_storage.tempfile,
            name='NamedTemporaryFile',
            value=Mock(side_effect=OSError('disk full')),
        )
        with pytest.raises(expected_exception=OSError, match='disk full'):
            write_json(path=tmp_path / 'report.json', value={})
        assert not list(tmp_path.iterdir())


class TestSustainedWindows:
    """Use a controlled clock to test long-window reporting without a minute-long test."""

    def test_progress_and_window_boundaries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify a first/last window split across two measured 31-second blocks.

        :param monkeypatch: Deterministic clock and inexpensive CPU telemetry.
        :type monkeypatch: pytest.MonkeyPatch
        """
        monkeypatch.setattr(
            target=benchmark.time,
            name='perf_counter',
            value=Mock(side_effect=[0.0, 0.0, 31.0, 31.0, 62.0]),
        )
        monkeypatch.setattr(target=benchmark, name='_cpu_frequency', value=Mock(return_value=None))
        runtime = Mock(batch_size=1)
        report = benchmark.measure_sustained(
            runtime=runtime,
            input_idx=np.empty(shape=0, dtype=np.int64),
            currents=np.empty(shape=(0, 1), dtype=np.float32),
            seconds=60,
        )
        assert runtime.run.call_count == 2
        assert report['elapsed_seconds'] == 62
        assert report['ms_per_episode_step_first_30s'] == pytest.approx(expected=3100)
        assert report['ms_per_episode_step_last_30s'] == pytest.approx(expected=3100)

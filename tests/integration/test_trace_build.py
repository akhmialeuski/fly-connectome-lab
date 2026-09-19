"""Check deterministic restart, batch invariance, memory modes, and trace commands."""

import json
from pathlib import Path
from unittest.mock import patch

import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from filelock import FileLock, Timeout
from typer.testing import CliRunner

from flystate.cli.main import app
from flystate.experiments.config import ExperimentConfig, effective_yaml
from flystate.hashing import sha256_file
from flystate.settings import get_paths
from flystate.storage.json import write_json
from flystate.traces.builder import build_trace, open_trace, resolve_trace, trace_stats
from flystate.traces.key import cache_key, cache_key_fields
from flystate.traces.store import TraceError, TraceStore


class TestTraceBuild:
    """Verify stored values across restart, execution batching, and state-reset policies."""

    def test_resume_and_batch_invariance(self, tiny_experiment: ExperimentConfig) -> None:
        """Resume a trial, compare fresh builds, then change batch size without changing values.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        trial = build_trace(cfg=tiny_experiment, paths=paths, max_chunks=1)
        assert trial['status'] == 'interrupted' and trial['chunks_done'] == 1
        with pytest.raises(expected_exception=TraceError, match='incomplete'):
            open_trace(cfg=tiny_experiment, paths=paths)
        result = build_trace(cfg=tiny_experiment, paths=paths)
        assert result['status'] == 'completed' and result['chunks_done'] == 3
        store = open_trace(cfg=tiny_experiment, paths=paths)
        features = np.asarray(a=store.array(name='features')[:])
        summaries = np.asarray(a=store.array(name='summary')[:])
        assert features.shape == (40, 4, 200) and features.dtype == np.float16
        before = (store.directory / 'build.json').read_bytes()
        assert build_trace(cfg=tiny_experiment, paths=paths)['key'] == result['key']
        assert (store.directory / 'build.json').read_bytes() == before
        assert build_trace(cfg=tiny_experiment, paths=paths, force=True)['key'] == result['key']
        np.testing.assert_array_equal(
            actual=open_trace(cfg=tiny_experiment, paths=paths).array(name='features')[:],
            desired=features,
        )
        single = tiny_experiment.model_copy(
            update={'brain': tiny_experiment.brain.model_copy(update={'batch_size': 1})}
        )
        assert build_trace(cfg=single, paths=paths, force=True)['key'] == result['key']
        rebuilt = open_trace(cfg=single, paths=paths)
        np.testing.assert_array_equal(actual=rebuilt.array(name='features')[:], desired=features)
        np.testing.assert_array_equal(actual=rebuilt.array(name='summary')[:], desired=summaries)
        stats = trace_stats(store=rebuilt)
        assert stats['counts_per_split'] == {'train': 28, 'val': 8, 'test': 4}
        np.testing.assert_allclose(
            actual=[row['mean_voltage'] for row in stats['observations']],
            desired=features[:, :, 100:].astype(np.float64).mean(axis=(0, 2)),
        )

    def test_memory_modes_and_key(self, tiny_experiment: ExperimentConfig) -> None:
        """Compare memory policies and verify numerical cache-key inputs.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        build_trace(cfg=tiny_experiment, paths=paths)
        persistent = np.asarray(
            a=open_trace(cfg=tiny_experiment, paths=paths).array(name='features')[:]
        )
        reset = tiny_experiment.model_copy(
            update={'memory': tiny_experiment.memory.model_copy(update={'mode': 'reset'})}
        )
        result = build_trace(cfg=reset, paths=paths)
        values = np.asarray(a=open_trace(cfg=reset, paths=paths).array(name='features')[:])
        np.testing.assert_array_equal(actual=persistent[:, 0], desired=values[:, 0])
        assert np.any(persistent[:, 1:] != values[:, 1:])
        concat = reset.model_copy(
            update={
                'name': 'concat',
                'memory': reset.memory.model_copy(update={'mode': 'reset_concat'}),
            }
        )
        assert build_trace(cfg=concat, paths=paths)['key'] == result['key']
        prepared, fields, _ = resolve_trace(cfg=tiny_experiment, paths=paths)
        other = tiny_experiment.model_copy(
            update={
                'name': 'other',
                'readout': tiny_experiment.readout.model_copy(update={'c_grid': (1.0,)}),
            }
        )
        same = cache_key_fields(
            cfg=other, prepared=prepared, brain_files_sha=fields['brain_files_sha256'], threads=1
        )
        assert same == fields
        changed = cache_key_fields(
            cfg=other, prepared=prepared, brain_files_sha=fields['brain_files_sha256'], threads=2
        )
        assert cache_key(fields=changed) != cache_key(fields=fields)
        with pytest.raises(expected_exception=ValueError):
            cache_key_fields(cfg=other, prepared=prepared, brain_files_sha={}, threads=0)

    def test_failures_and_lock(self, tiny_experiment: ExperimentConfig) -> None:
        """Persist failures, restore threads, exclude a second writer, and resume successfully.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        original = numba.get_num_threads()
        with (
            patch(target='flystate.traces.builder._simulate_chunk', side_effect=KeyboardInterrupt),
            pytest.raises(expected_exception=KeyboardInterrupt),
        ):
            build_trace(cfg=tiny_experiment, paths=paths)
        _, _, directory = resolve_trace(cfg=tiny_experiment, paths=paths)
        assert TraceStore(directory=directory).meta['status'] == 'interrupted'
        assert numba.get_num_threads() == original
        with (
            patch(
                target='flystate.traces.builder._simulate_chunk', side_effect=RuntimeError('failed')
            ),
            pytest.raises(expected_exception=RuntimeError),
        ):
            build_trace(cfg=tiny_experiment, paths=paths)
        assert TraceStore(directory=directory).meta['status'] == 'failed'
        with (
            FileLock(lock_file=paths.features / '.locks' / f'{directory.name}.lock', timeout=0),
            pytest.raises(expected_exception=Timeout),
        ):
            build_trace(cfg=tiny_experiment, paths=paths, force=True)
        assert directory.exists()
        assert build_trace(cfg=tiny_experiment, paths=paths)['status'] == 'completed'
        with pytest.raises(expected_exception=TraceError):
            build_trace(cfg=tiny_experiment, paths=paths, max_chunks=0)

    def test_cli(self, tiny_experiment: ExperimentConfig, tmp_path: Path) -> None:
        """Exercise all commands, overrides, JSON failures, and interruption exit status.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        :param tmp_path: Configuration directory.
        :type tmp_path: Path
        """
        config = tmp_path / 'tiny.yaml'
        config.write_text(data=effective_yaml(cfg=tiny_experiment))
        runner = CliRunner()
        for command in ('build', 'inspect', 'stats'):
            result = runner.invoke(
                app=app,
                args=['trace', command, str(config), '--set', 'brain.batch_size=1', '--json'],
            )
            assert result.exit_code == 0, result.output
            assert len(json.loads(s=result.stdout)['key']) == 64
        text = runner.invoke(app=app, args=['trace', 'inspect', str(config)])
        assert text.exit_code == 0 and 'completed' in text.stdout
        bad = runner.invoke(app=app, args=['trace', 'build', str(tmp_path / 'missing'), '--json'])
        assert bad.exit_code == 2
        with patch(target='flystate.cli.trace.build_trace', side_effect=KeyboardInterrupt):
            result = runner.invoke(app=app, args=['trace', 'build', str(config), '--json'])
        assert result.exit_code == 130 and json.loads(s=result.stdout)['interrupted']
        with patch(target='flystate.cli.trace.open_trace', side_effect=TraceError('corrupt')):
            result = runner.invoke(app=app, args=['trace', 'inspect', str(config), '--json'])
        assert result.exit_code == 1
        with patch(target='flystate.cli.trace.open_trace', side_effect=TraceError('corrupt')):
            result = runner.invoke(app=app, args=['trace', 'inspect', str(config)])
        assert result.exit_code == 1 and result.stdout == ''

    def test_index_identity(self, tiny_experiment: ExperimentConfig) -> None:
        """Reject changed labels even if the index file's own digest is updated.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        paths = get_paths()
        build_trace(cfg=tiny_experiment, paths=paths, max_chunks=1)
        _, _, directory = resolve_trace(cfg=tiny_experiment, paths=paths)
        store = TraceStore(directory=directory)
        store.index[0]['label'] = 999
        pq.write_table(
            table=pa.Table.from_pylist(mapping=store.index), where=directory / 'index.parquet'
        )
        store.meta['index_sha256'] = sha256_file(path=directory / 'index.parquet')
        write_json(path=directory / 'build.json', value=store.meta)
        with pytest.raises(expected_exception=TraceError, match='index'):
            open_trace(cfg=tiny_experiment, paths=paths)
        with pytest.raises(expected_exception=TraceError, match='index'):
            build_trace(cfg=tiny_experiment, paths=paths)

    @pytest.mark.parametrize('error', [RuntimeError('initialization failed'), KeyboardInterrupt()])
    def test_initialization_failure(
        self, tiny_experiment: ExperimentConfig, error: BaseException
    ) -> None:
        """Restore Numba settings when construction fails before a store exists.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        :param error: Initialization failure or interruption.
        :type error: BaseException
        """
        original = numba.get_num_threads()
        with (
            patch(target='flystate.traces.builder.EpisodeBrain', side_effect=error),
            pytest.raises(expected_exception=type(error)),
        ):
            build_trace(cfg=tiny_experiment, paths=get_paths())
        assert numba.get_num_threads() == original

    def test_progress_during_chunk(self, tiny_experiment: ExperimentConfig) -> None:
        """Emit throughput while batches are being simulated, before the chunk is committed.

        :param tiny_experiment: Fully installed offline experiment.
        :type tiny_experiment: ExperimentConfig
        """
        with (
            patch(target='flystate.traces.builder.PROGRESS_SECONDS', new=0),
            patch(target='flystate.traces.builder._progress') as progress,
        ):
            build_trace(cfg=tiny_experiment, paths=get_paths(), max_chunks=1)
        assert progress.call_count > 2
        assert progress.call_args_list[0].kwargs['processed'] == 3

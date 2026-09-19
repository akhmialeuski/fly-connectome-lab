"""Test transactional chunk publication, corruption checks, and immutable completion."""

from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flystate.datasets.subset import Sample
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file
from flystate.storage.json import write_json
from flystate.traces.builder import _progress, trace_stats
from flystate.traces.key import cache_key
from flystate.traces.store import TraceError, TraceStore


def make_store(root: Path, kinds: list[str] | None = None) -> TraceStore:
    """Create a three-row cache for storage-only tests.

    :param root: Isolated parent directory.
    :type root: Path
    :param kinds: Optional readout feature names.
    :type kinds: Optional[list[str]]
    :returns: Writable empty trace store.
    :rtype: TraceStore
    """
    fields = {'format': 1, 'readout': {'features': kinds or ['spike_trace', 'voltage']}}
    samples = [
        Sample(sample_id=f's{i}', filename=f'{i:06}.jpg', identity=i, label=i, split='train')
        for i in range(3)
    ]
    return TraceStore.create(
        directory=root / cache_key(fields=fields),
        fields=fields,
        samples=samples,
        steps=2,
        features=4,
        threads=1,
        batch_size=1,
    )


def write_payload(store: TraceStore) -> None:
    """Publish one deterministic valid test chunk.

    :param store: Empty three-row test store.
    :type store: TraceStore
    """
    store.write_chunk(
        chunk=0,
        features=np.arange(24).reshape(3, 2, 4),
        summary=np.ones(shape=(3, 2, 3)),
        brain_ms=np.ones(shape=3),
    )


class TestTraceStore:
    """Exercise file-level integrity and completion contracts independently of simulation."""

    def test_commit_and_immutability(self, tmp_path: Path) -> None:
        """Commit arrays, verify content, and reject mutations after completion or through readers.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        """
        store = make_store(root=tmp_path)
        with pytest.raises(expected_exception=TraceError, match='missing chunks'):
            store.set_status(status='completed')
        with pytest.raises(expected_exception=TraceError, match='status'):
            store.set_status(status='unknown')
        write_payload(store=store)
        with pytest.raises(expected_exception=TraceError, match='overwritten'):
            write_payload(store=store)
        with pytest.raises(expected_exception=TraceError, match='not completed'):
            store.verify(require_complete=True)
        store.set_status(status='completed')
        assert store.verify(require_complete=True).all()
        with pytest.raises(expected_exception=TraceError, match='read-only'):
            store.set_status(status='running')
        reader = TraceStore(directory=store.directory)
        with pytest.raises(expected_exception=TraceError, match='read-only'):
            write_payload(store=reader)
        with pytest.raises(expected_exception=TraceError, match='range'):
            store.bounds(chunk=2)

    @pytest.mark.parametrize(
        'bad', ['shape', 'nan', 'overflow', 'negative_counts', 'negative_time']
    )
    def test_invalid_payload(self, tmp_path: Path, bad: str) -> None:
        """Reject invalid payloads before any completion marker is visible.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        :param bad: Invalid feature, count, or timing category.
        :type bad: str
        """
        store = make_store(root=tmp_path)
        features = np.ones(shape=(3, 2, 4))
        summary = np.ones(shape=(3, 2, 3))
        timings = np.ones(shape=3)
        if bad == 'shape':
            features = features[:1]
        elif bad == 'nan':
            features[0, 0, 0] = np.nan
        elif bad == 'overflow':
            features[0, 0, 0] = 1e10
        elif bad == 'negative_counts':
            summary[0, 0, 0] = -1
        else:
            timings[0] = -1
        with pytest.raises(expected_exception=TraceError):
            store.write_chunk(chunk=0, features=features, summary=summary, brain_ms=timings)
        assert not store.verify().any()

    def test_partial_publication(self, tmp_path: Path) -> None:
        """Interrupt the commit record write, then verify safe overwrite of the uncommitted payload.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        """
        store = make_store(root=tmp_path)
        with (
            patch(target='flystate.traces.store.write_json', side_effect=KeyboardInterrupt),
            pytest.raises(expected_exception=KeyboardInterrupt),
        ):
            write_payload(store=store)
        assert not store.verify().any()
        write_payload(store=store)
        assert store.verify().all()

    @pytest.mark.parametrize('bad', ['features', 'record', 'missing_record', 'completed_mask'])
    def test_committed_corruption(self, tmp_path: Path, bad: str) -> None:
        """Reject altered committed arrays, digests, or completion masks.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        :param bad: Corrupted committed artifact.
        :type bad: str
        """
        store = make_store(root=tmp_path)
        write_payload(store=store)
        if bad == 'features':
            store.array(name='features')[0, 0, 0] = 100
        elif bad == 'record':
            (store.directory / 'chunks' / '0.json').write_text(data='{}')
        elif bad == 'missing_record':
            (store.directory / 'chunks' / '0.json').unlink()
        else:
            store.set_status(status='completed')
            store.array(name='done')[0] = False
        with pytest.raises(expected_exception=TraceError):
            store.verify()

    @pytest.mark.parametrize(
        'bad',
        [
            'key',
            'dimensions',
            'schema',
            'group',
            'index_digest',
            'index_rows',
            'index_duplicates',
            'status',
            'missing_metadata',
        ],
    )
    def test_metadata_corruption(self, tmp_path: Path, bad: str) -> None:
        """Reject malformed metadata and structural changes even with refreshed index digests.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        :param bad: Metadata or schema fault.
        :type bad: str
        """
        store = make_store(root=tmp_path)
        meta = store.meta.copy()
        if bad == 'key':
            meta['key'] = 'invalid'
        elif bad == 'dimensions':
            meta['N'] = 0
        elif bad in {'schema', 'group'}:
            del store.group['features']
            if bad == 'group':
                store.group.create_group(name='features')
            else:
                store.group.create_array(name='features', shape=(1,), dtype=np.float32)
        elif bad == 'index_digest':
            meta['index_sha256'] = 'invalid'
        elif bad in {'index_rows', 'index_duplicates'}:
            rows = store.index
            if bad == 'index_rows':
                rows[0]['row'] = 2
            else:
                rows[0]['sample_id'] = rows[1]['sample_id']
            pq.write_table(
                table=pa.Table.from_pylist(mapping=rows), where=store.directory / 'index.parquet'
            )
            meta['index_sha256'] = sha256_file(path=store.directory / 'index.parquet')
        elif bad == 'status':
            meta['status'] = 'unknown'
        else:
            (store.directory / 'build.json').unlink()
        if bad != 'missing_metadata':
            write_json(path=store.directory / 'build.json', value=meta)
        with pytest.raises(expected_exception=TraceError):
            TraceStore(directory=store.directory)

    def test_atomic_initialization(self, tmp_path: Path) -> None:
        """An interrupted initialization leaves no published unusable cache or staging files.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        """
        with (
            patch(target='flystate.traces.store.write_json', side_effect=KeyboardInterrupt),
            pytest.raises(expected_exception=KeyboardInterrupt),
        ):
            make_store(root=tmp_path)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize('kind', ['voltage', 'spike_trace'])
    def test_single_feature_stats(self, tmp_path: Path, kind: str) -> None:
        """Report absent feature families as null, without interpreting unrelated columns.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        :param kind: Single stored feature family.
        :type kind: str
        """
        store = make_store(root=tmp_path, kinds=[kind])
        write_payload(store=store)
        store.set_status(status='completed')
        report = trace_stats(store=store)
        missing = 'mean_voltage' if kind == 'spike_trace' else 'mean_spike_trace'
        assert report['observations'][0][missing] is None

    def test_progress_window(self, tmp_path: Path) -> None:
        """Trim stale rate history and preserve a null ETA before any sample completes.

        :param tmp_path: Isolated artifact root.
        :type tmp_path: Path
        """
        store = make_store(root=tmp_path)
        history = deque([(0.0, 0), (1.0, 1), (400.0, 2), (401.0, 3)])
        _progress(
            store=store,
            history=history,
            started=0,
            processed=3,
            cfg=ExperimentConfig(name='progress'),
        )
        assert len(history) == 3
        with patch(target='flystate.traces.builder.get_logger') as logger:
            _progress(
                store=store,
                history=deque([(0.0, 0)]),
                started=0,
                processed=0,
                cfg=ExperimentConfig(name='progress'),
            )
        assert logger.return_value.info.call_args.kwargs['eta_seconds'] is None

"""Zarr v3 storage with explicit chunk commits and content-integrity validation."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import flybrain
import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from numpy.typing import NDArray

from flystate import __version__
from flystate.datasets.subset import Sample
from flystate.hashing import sha256_file
from flystate.storage.json import write_json
from flystate.traces.key import cache_key

CHUNK_SIZE: int = 64
DATA_ARRAYS: tuple[str, ...] = ('features', 'summary', 'brain_ms')


class TraceError(ValueError):
    """A trace cache is missing, incomplete, incompatible, or corrupt."""


def index_rows(samples: list[Sample]) -> list[dict[str, Any]]:
    """Represent canonical trace row identities without source image pixels.

    :param samples: Ordered dataset samples.
    :type samples: list[Sample]
    :returns: Parquet-compatible row records.
    :rtype: list[dict[str, Any]]
    """
    return [
        {
            'row': row,
            'sample_id': sample.sample_id,
            'split': sample.split,
            'label': sample.label,
            'identity': sample.identity,
        }
        for row, sample in enumerate(samples)
    ]


def directory_size(directory: Path) -> int:
    """Measure allocated artifact file lengths without loading their contents.

    :param directory: Existing storage root.
    :type directory: Path
    :returns: Sum of file sizes in bytes.
    :rtype: int
    """
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())


class TraceStore:
    """Validated Zarr arrays with a JSON commit record for every complete chunk."""

    def __init__(self, directory: Path, mode: Literal['r', 'a'] = 'r') -> None:
        """Open existing metadata and reject incompatible array or index layouts.

        :param directory: Existing cache directory.
        :type directory: Path
        :param mode: Read-only inspection or locked writer access.
        :type mode: Literal['r', 'a']
        :raises TraceError: If metadata, schemas, or the index digest are invalid.
        """
        self.directory: Path = directory
        self.mode = mode
        try:
            self.meta: dict[str, Any] = json.loads(
                s=(directory / 'build.json').read_text(encoding='utf-8')
            )
            if (
                self.meta['key'] != cache_key(fields=self.meta['key_fields'])
                or directory.name != self.meta['key']
            ):
                raise TraceError('Trace key does not match its identity fields or directory.')
            self.n: int = self.meta['N']
            self.steps: int = self.meta['T']
            self.features_count: int = self.meta['F']
            self.chunk_size: int = self.meta['chunk_size']
            if min(self.n, self.steps, self.features_count, self.chunk_size) < 1:
                raise TraceError('Trace array dimensions must be positive.')
            self.chunks: int = (self.n + self.chunk_size - 1) // self.chunk_size
            self.group: zarr.Group = zarr.open_group(store=directory / 'features.zarr', mode=mode)
            for name, shape, dtype in (
                ('features', (self.n, self.steps, self.features_count), np.float16),
                ('summary', (self.n, self.steps, 3), np.float32),
                ('brain_ms', (self.n,), np.float32),
                ('done', (self.chunks,), np.bool_),
            ):
                array = self.array(name=name)
                expected_chunks = (1,) if name == 'done' else (self.chunk_size, *shape[1:])
                if (
                    array.shape != shape
                    or array.dtype != np.dtype(dtype)
                    or array.chunks != expected_chunks
                ):
                    raise TraceError(f'Invalid trace array schema: {name}.')
            if sha256_file(path=directory / 'index.parquet') != self.meta['index_sha256']:
                raise TraceError('Trace index integrity check failed.')
            self.index: list[dict[str, Any]] = pq.read_table(
                source=directory / 'index.parquet'
            ).to_pylist()
            if len(self.index) != self.n or [row['row'] for row in self.index] != list(
                range(self.n)
            ):
                raise TraceError('Trace index row order or count is invalid.')
            if len({row['sample_id'] for row in self.index}) != self.n:
                raise TraceError('Trace sample identifiers must be unique.')
            if self.meta['status'] not in {'running', 'interrupted', 'failed', 'completed'}:
                raise TraceError('Unknown trace build status.')
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise TraceError(f'Cannot open trace cache {directory}: {error}') from error

    @classmethod
    def create(
        cls,
        directory: Path,
        fields: dict[str, Any],
        samples: list[Sample],
        steps: int,
        features: int,
        threads: int,
        batch_size: int,
    ) -> 'TraceStore':
        """Initialize an empty store; the caller must hold its external writer lock.

        :param directory: New cache directory named by its key.
        :type directory: Path
        :param fields: Resolved cache identity fields.
        :type fields: dict[str, Any]
        :param samples: Canonical sample order.
        :type samples: list[Sample]
        :param steps: Observation count.
        :type steps: int
        :param features: Readout feature dimension.
        :type features: int
        :param threads: Resolved simulation thread count.
        :type threads: int
        :param batch_size: Execution batch size for the first attempt.
        :type batch_size: int
        :returns: Writable validated store.
        :rtype: TraceStore
        """
        directory.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f'.{directory.name}.', dir=directory.parent) as staging:
            temporary = Path(staging)
            group = zarr.open_group(store=temporary / 'features.zarr', mode='w', zarr_format=3)
            n = len(samples)
            for name, shape, dtype in (
                ('features', (n, steps, features), np.float16),
                ('summary', (n, steps, 3), np.float32),
                ('brain_ms', (n,), np.float32),
            ):
                group.create_array(
                    name=name,
                    shape=shape,
                    chunks=(CHUNK_SIZE, *shape[1:]),
                    dtype=dtype,
                    fill_value=np.nan,
                )
            group.create_array(
                name='done',
                shape=((n + CHUNK_SIZE - 1) // CHUNK_SIZE,),
                chunks=(1,),
                dtype=np.bool_,
                fill_value=False,
            )
            pq.write_table(
                table=pa.Table.from_pylist(mapping=index_rows(samples=samples)),
                where=temporary / 'index.parquet',
            )
            now = datetime.now(tz=UTC).isoformat()
            meta = {
                'key': cache_key(fields=fields),
                'key_fields': fields,
                'status': 'running',
                'created_utc': now,
                'updated_utc': now,
                'threads': threads,
                'batch_size': batch_size,
                'N': n,
                'T': steps,
                'F': features,
                'chunk_size': CHUNK_SIZE,
                'index_sha256': sha256_file(path=temporary / 'index.parquet'),
                'versions': {
                    'flystate': __version__,
                    'flybrain': flybrain.__version__,
                    'numba': numba.__version__,
                    'zarr': zarr.__version__,
                    'numpy': np.__version__,
                },
            }
            write_json(path=temporary / 'build.json', value=meta)
            temporary.rename(target=directory)
        return cls(directory=directory, mode='a')

    def array(self, name: str) -> zarr.Array:
        """Resolve a numeric array and reject a group at the requested path.

        :param name: Required array basename.
        :type name: str
        :returns: Typed Zarr array.
        :rtype: zarr.Array
        :raises TraceError: If the member is a group instead of an array.
        """
        array = self.group[name]
        if not isinstance(array, zarr.Array):
            raise TraceError(f'Trace member is not an array: {name}.')
        return array

    def bounds(self, chunk: int) -> tuple[int, int]:
        """Return the half-open row interval belonging to one chunk.

        :param chunk: Zero-based chunk index.
        :type chunk: int
        :returns: Inclusive start and exclusive stop row.
        :rtype: tuple[int, int]
        :raises TraceError: If the chunk index is out of range.
        """
        if not 0 <= chunk < self.chunks:
            raise TraceError('Trace chunk index is out of range.')
        return chunk * self.chunk_size, min((chunk + 1) * self.chunk_size, self.n)

    def verify(self, require_complete: bool = False) -> NDArray[np.bool_]:
        """Validate each committed chunk before resuming or exposing its features.

        :param require_complete: Require completed status and all chunk markers.
        :type require_complete: bool
        :returns: Verified completion mask, shape (chunks,).
        :rtype: NDArray[np.bool_]
        :raises TraceError: If a committed chunk is corrupt or required work is incomplete.
        """
        done = np.asarray(a=self.array(name='done')[:], dtype=np.bool_)
        for chunk in np.flatnonzero(a=done):
            start, stop = self.bounds(chunk=int(chunk))
            try:
                hashes = json.loads(s=(self.directory / 'chunks' / f'{chunk}.json').read_text())
                for name in DATA_ARRAYS:
                    values = np.asarray(a=self.array(name=name)[start:stop])
                    digest = hashlib.sha256(string=values.tobytes()).hexdigest()
                    if digest != hashes[name] or not np.isfinite(values).all():
                        raise TraceError(f'Trace chunk {chunk} failed integrity check: {name}.')
            except (OSError, KeyError, ValueError) as error:
                raise TraceError(f'Invalid committed trace chunk {chunk}: {error}') from error
        if (require_complete or self.meta['status'] == 'completed') and not done.all():
            raise TraceError('Trace is incomplete; run flystate trace build CONFIG.')
        if require_complete and self.meta['status'] != 'completed':
            raise TraceError('Trace is not completed; run flystate trace build CONFIG.')
        return done

    def write_chunk(
        self, chunk: int, features: NDArray, summary: NDArray, brain_ms: NDArray
    ) -> None:
        """Write numeric payloads and integrity records before publishing the done marker.

        :param chunk: Zero-based uncommitted chunk index.
        :type chunk: int
        :param features: Float-convertible features (rows,T,F).
        :type features: NDArray
        :param summary: Observation spike counts (rows,T,3).
        :type summary: NDArray
        :param brain_ms: Nonnegative episode simulation times (rows,), milliseconds.
        :type brain_ms: NDArray
        :raises TraceError: If the store is immutable or a payload is malformed/nonfinite.
        """
        self._writable()
        start, stop = self.bounds(chunk=chunk)
        if self.array(name='done')[chunk]:
            raise TraceError('A committed trace chunk cannot be overwritten.')
        hashes = {}
        payloads = {}
        for name, value in zip(DATA_ARRAYS, (features, summary, brain_ms), strict=True):
            array = self.array(name=name)
            with np.errstate(over='ignore', invalid='ignore'):
                converted = np.asarray(a=value, dtype=array.dtype)
            if (
                converted.shape != (stop - start, *array.shape[1:])
                or not np.isfinite(converted).all()
            ):
                raise TraceError(f'Invalid finite trace chunk payload: {name}.')
            if name != 'features' and np.any(converted < 0):
                raise TraceError(f'Trace counts and timings must be nonnegative: {name}.')
            hashes[name] = hashlib.sha256(string=converted.tobytes()).hexdigest()
            payloads[name] = converted
        for name, payload in payloads.items():
            self.array(name=name)[start:stop] = payload
        write_json(path=self.directory / 'chunks' / f'{chunk}.json', value=hashes)
        self.array(name='done')[chunk] = True

    def _writable(self) -> None:
        """Reject mutation through read-only handles or completed stores.

        :raises TraceError: If the store is read-only or completed.
        """
        if self.mode != 'a' or self.meta['status'] == 'completed':
            raise TraceError('Trace store is read-only or already completed.')

    def set_status(self, status: str, **fields: Any) -> None:
        """Atomically persist attempt state while preserving completed store immutability.

        :param status: New valid build state.
        :type status: str
        :param fields: Additional attempt metadata.
        :type fields: Any
        :raises TraceError: If the handle or requested state transition is invalid.
        """
        self._writable()
        if status not in {'running', 'interrupted', 'failed', 'completed'}:
            raise TraceError('Unknown trace build status.')
        if status == 'completed' and not self.verify().all():
            raise TraceError('Cannot complete a trace with missing chunks.')
        self.meta.update(fields)
        self.meta.update(status=status, updated_utc=datetime.now(tz=UTC).isoformat())
        write_json(path=self.directory / 'build.json', value=self.meta)

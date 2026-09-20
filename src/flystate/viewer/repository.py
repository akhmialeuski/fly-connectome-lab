"""Read-only artifact access with bounded queries and explicit storage boundaries."""

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from flystate.episodes.episode import EpisodeBuilder
from flystate.experiments.config import ConfigError, load_config
from flystate.hashing import sha256_file
from flystate.settings import Paths
from flystate.storage.parquet import read_table
from flystate.storage.runs import load_manifest
from flystate.traces.key import cache_key
from flystate.traces.store import DATA_ARRAYS, TraceStore

REPORT_KINDS: tuple[str, ...] = (
    'comparisons',
    'calibrations',
    'benchmarks',
    'design-checks',
    'reports',
)


def resource(root: Path, *parts: str) -> Path:
    """Resolve artifact identifiers while rejecting traversal and escaping symlinks.

    :param root: Allowed storage boundary.
    :type root: Path
    :param parts: Basenames identifying descendants.
    :type parts: str
    :returns: Checked resolved descendant.
    :rtype: Path
    :raises ValueError: If an identifier or resolved boundary is invalid.
    :raises FileNotFoundError: If the resource is absent.
    """
    if any(
        part in {'.', '..'} or not re.fullmatch(pattern=r'[A-Za-z0-9_.-]+', string=part)
        for part in parts
    ):
        raise ValueError('Invalid artifact identifier.')
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes its storage directory.')
    if not target.exists():
        raise FileNotFoundError('Requested artifact is unavailable.')
    return target


def document(path: Path) -> dict[str, Any]:
    """Read one finite JSON object from an artifact.

    :param path: Existing JSON file.
    :type path: Path
    :returns: Parsed object.
    :rtype: dict[str, Any]
    :raises ValueError: If JSON is not an object or contains nonfinite values.
    """
    value = json.loads(s=path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Artifact must contain a JSON object.')
    json.dumps(obj=value, allow_nan=False)
    return value


class Repository:
    """Expose recorded results without building caches, fitting models, or writing files."""

    def __init__(self, paths: Paths) -> None:
        """Bind the storage installation and an instance-local integrity memo.

        :param paths: Local experiment storage paths.
        :type paths: Paths
        """
        self.paths = paths
        self._verified: set[tuple[str, int, int, str]] = set()

    def verify(self, path: Path, expected: str) -> None:
        """Check content hashes once per observed file revision.

        :param path: Existing immutable artifact.
        :type path: Path
        :param expected: Recorded SHA-256 digest.
        :type expected: str
        :raises ValueError: If artifact content differs from its recorded digest.
        """
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, expected)
        if identity not in self._verified:
            if sha256_file(path=path) != expected:
                raise ValueError('Artifact integrity verification failed.')
            self._verified.add(identity)

    def catalog(self) -> dict[str, Any]:
        """List valid runs, reports, and trace status while surfacing damaged entries.

        :returns: Discovery records and recoverable artifact warnings.
        :rtype: dict[str, Any]
        """
        runs, reports, caches, warnings = [], [], [], []
        for source in sorted(self.paths.runs.glob('*/manifest.json'), reverse=True):
            try:
                directory = resource(self.paths.runs, source.parent.name)
                manifest = load_manifest(run_dir=resource(directory, 'manifest.json').parent)
                cfg = load_config(path=resource(directory, 'config.yaml'))
                summary_path = directory / 'summary.json'
                runs.append(
                    {
                        **manifest,
                        'name': cfg.name,
                        'mode': cfg.memory.mode,
                        'classes': cfg.dataset.subset.n_identities,
                        'steps': cfg.episodes.steps,
                        'summary': document(path=resource(directory, 'summary.json'))
                        if summary_path.exists()
                        else {},
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, ConfigError) as error:
                warnings.append({'artifact': source.parent.name, 'error': str(error)})
        for kind in REPORT_KINDS:
            parent = self.paths.runs / kind
            for source in sorted(parent.glob('*'), reverse=True):
                if source.suffix not in {'.json', '.md'} or source.name == 'latest.json':
                    continue
                try:
                    path = resource(self.paths.runs, kind, source.name)
                    payload = document(path=path) if path.suffix == '.json' else {}
                    reports.append(
                        {
                            'kind': kind,
                            'id': source.name,
                            'created_utc': payload.get('created_utc'),
                            'name': payload.get('config_name')
                            or payload.get('run_a')
                            or source.stem,
                        }
                    )
                except (OSError, ValueError, KeyError, TypeError, ConfigError) as error:
                    warnings.append({'artifact': source.name, 'error': str(error)})
        for source in sorted(self.paths.features.glob('*/build.json')):
            try:
                path = resource(self.paths.features, source.parent.name, 'build.json')
                meta = document(path=path)
                caches.append(
                    {
                        key: meta.get(key)
                        for key in ('key', 'status', 'N', 'T', 'F', 'threads', 'batch_size')
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, ConfigError) as error:
                warnings.append({'artifact': source.parent.name, 'error': str(error)})
        return {'runs': runs, 'reports': reports, 'caches': caches, 'warnings': warnings}

    def run(self, run_id: str) -> dict[str, Any]:
        """Read a run's provenance, validation curve, and evaluation inventory.

        :param run_id: Run directory basename.
        :type run_id: str
        :returns: Run detail and available evaluations.
        :rtype: dict[str, Any]
        """
        directory = resource(self.paths.runs, run_id)
        result: dict[str, Any] = {
            'manifest': load_manifest(run_dir=resource(directory, 'manifest.json').parent),
            'config': load_config(path=resource(directory, 'config.yaml')).model_dump(mode='json'),
            'evaluations': [],
            'warnings': [],
        }
        try:
            result['identities'] = self.identity_references(manifest=result['manifest'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['identities'] = []
            result['warnings'].append(f'Identity reference images unavailable: {error}')
        for name, parts in {
            'summary': ('summary.json',),
            'environment': ('environment.json',),
            'model': ('model', 'model.json'),
        }.items():
            if directory.joinpath(*parts).exists():
                result[name] = document(path=resource(directory, *parts))
        validation = directory / 'metrics/validation.parquet'
        result['validation'] = (
            read_table(path=resource(directory, 'metrics', 'validation.parquet'))
            if validation.exists()
            else []
        )
        for source in sorted((directory / 'evals').glob('*/eval.json'), reverse=True):
            try:
                meta = document(path=resource(directory, 'evals', source.parent.name, 'eval.json'))
                if meta.get('run_id') != run_id or meta.get('eval_id') != source.parent.name:
                    raise ValueError('Evaluation identity does not match its directory.')
                result['evaluations'].append(meta)
            except (OSError, ValueError) as error:
                result['warnings'].append(str(error))
        return result

    def identity_references(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        """Map output classes to deterministic training-only example photographs.

        :param manifest: Recorded run identity and trace-cache provenance.
        :type manifest: dict[str, Any]
        :returns: One training sample and original dataset identity per output label.
        :rtype: list[dict[str, Any]]
        :raises ValueError: If the cache identity differs from the recorded run.
        """
        directory = resource(self.paths.features, manifest['cache_key'])
        meta = document(path=resource(directory, 'build.json'))
        if (
            meta['key'] != manifest['cache_key']
            or cache_key(fields=meta['key_fields']) != manifest['cache_key']
            or meta['key_fields']['dataset_fingerprint'] != manifest['dataset_fingerprint']
        ):
            raise ValueError('Identity reference cache differs from the recorded run.')
        index = resource(directory, 'index.parquet')
        self.verify(path=index, expected=meta['index_sha256'])
        references: dict[int, dict[str, Any]] = {}
        for row in read_table(path=index):
            if row['split'] == 'train':
                references.setdefault(
                    row['label'],
                    {
                        'label': row['label'],
                        'identity': row['identity'],
                        'sample_id': row['sample_id'],
                    },
                )
        return [references[label] for label in sorted(references)]

    def evaluation_directory(self, run_id: str, eval_id: str) -> Path:
        """Resolve a completed evaluation belonging to the requested run.

        :param run_id: Run directory basename.
        :type run_id: str
        :param eval_id: Evaluation directory basename.
        :type eval_id: str
        :returns: Checked evaluation directory.
        :rtype: Path
        :raises ValueError: If status or recorded identity is inconsistent.
        """
        directory = resource(self.paths.runs, run_id, 'evals', eval_id)
        meta = document(path=resource(directory, 'eval.json'))
        if (
            meta.get('status') != 'completed'
            or meta.get('run_id') != run_id
            or meta.get('eval_id') != eval_id
        ):
            raise ValueError('Evaluation is incomplete or has inconsistent provenance.')
        return directory

    def evaluation(self, run_id: str, eval_id: str) -> dict[str, Any]:
        """Read evaluation metrics, sparse confusion counts, and per-image timings.

        :param run_id: Run directory basename.
        :type run_id: str
        :param eval_id: Evaluation directory basename.
        :type eval_id: str
        :returns: Evaluation metadata and display tables.
        :rtype: dict[str, Any]
        """
        directory = self.evaluation_directory(run_id=run_id, eval_id=eval_id)
        return {
            'meta': document(path=resource(directory, 'eval.json')),
            **{
                name: read_table(path=resource(directory, f'{name}.parquet'))
                for name in ('metrics', 'confusion', 'timings')
            },
        }

    def predictions(
        self,
        run_id: str,
        eval_id: str,
        t: int | None,
        correct: bool | None,
        sample_id: str | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        """Return a bounded page of predictions filtered at the Parquet read boundary.

        :param run_id: Run directory basename.
        :type run_id: str
        :param eval_id: Evaluation directory basename.
        :type eval_id: str
        :param t: Optional one-based observation filter.
        :type t: Optional[int]
        :param correct: Optional correctness filter.
        :type correct: Optional[bool]
        :param sample_id: Optional exact sample identity.
        :type sample_id: Optional[str]
        :param offset: Nonnegative page offset.
        :type offset: int
        :param limit: Page size from 1 to 200.
        :type limit: int
        :returns: Page records, total matching count, and pagination settings.
        :rtype: dict[str, Any]
        """
        directory = self.evaluation_directory(run_id=run_id, eval_id=eval_id)
        filters = [
            (name, '=', value)
            for name, value in (('t', t), ('correct', correct), ('sample_id', sample_id))
            if value is not None
        ]
        table = pq.read_table(
            source=resource(directory, 'predictions.parquet'), filters=filters or None
        )
        return {
            'rows': table.slice(offset=offset, length=limit).to_pylist(),
            'total': table.num_rows,
            'offset': offset,
            'limit': limit,
        }

    def report(self, kind: str, name: str) -> dict[str, Any]:
        """Read one known report category as JSON or plain Markdown text.

        :param kind: Supported report category.
        :type kind: str
        :param name: Report filename.
        :type name: str
        :returns: Report payload without interpreting embedded HTML.
        :rtype: dict[str, Any]
        :raises ValueError: If category or extension is unsupported.
        """
        if kind not in REPORT_KINDS or Path(name).suffix not in {'.json', '.md'}:
            raise ValueError('Unsupported report resource.')
        path = resource(self.paths.runs, kind, name)
        return (
            {'text': path.read_text(encoding='utf-8')}
            if path.suffix == '.md'
            else document(path=path)
        )

    def sample(self, run_id: str, sample_id: str) -> dict[str, Any]:
        """Read one cached episode with feature blocks and exact stimulus geometry.

        :param run_id: Run directory basename.
        :type run_id: str
        :param sample_id: Recorded sample identity.
        :type sample_id: str
        :returns: Episode metadata, feature matrices (T,R), and optional neuron geometry.
        :rtype: dict[str, Any]
        :raises FileNotFoundError: If the sample is absent from the recorded cache.
        :raises ValueError: If cache provenance or dimensions disagree with the run.
        """
        directory = resource(self.paths.runs, run_id)
        manifest = load_manifest(run_dir=resource(directory, 'manifest.json').parent)
        cfg = load_config(path=resource(directory, 'config.yaml'))
        cache = resource(self.paths.features, manifest['cache_key'])
        # Zarr opens internal descendants itself; check symlinks before handing it the tree.
        for member in cache.rglob('*'):
            if member.is_symlink() and not member.resolve().is_relative_to(cache):
                raise ValueError('Trace member escapes its storage directory.')
        store = TraceStore(directory=cache)
        if store.meta['key_fields']['dataset_fingerprint'] != manifest['dataset_fingerprint']:
            raise ValueError('Trace dataset identity differs from the run.')
        matching = [row for row in store.index if row['sample_id'] == sample_id]
        if not matching:
            raise FileNotFoundError('Sample does not belong to this run.')
        sample = matching[0]
        chunk = sample['row'] // store.chunk_size
        if not bool(store.array(name='done')[chunk]):
            raise ValueError('Sample trace chunk is not completed.')
        start, stop = store.bounds(chunk=chunk)
        hashes = document(path=resource(store.directory, 'chunks', f'{chunk}.json'))
        for name in DATA_ARRAYS:
            payload = np.asarray(a=store.array(name=name)[start:stop])
            if hashlib.sha256(string=payload.tobytes()).hexdigest() != hashes[name]:
                raise ValueError(f'Trace chunk integrity verification failed: {name}.')
        values = np.asarray(a=store.array(name='features')[sample['row']], dtype=np.float32)
        if not np.isfinite(values).all() or values.shape[1] % len(cfg.readout.features):
            raise ValueError('Trace features have invalid values or geometry.')
        blocks = np.split(ary=values, indices_or_sections=len(cfg.readout.features), axis=1)
        neurons: dict[str, Any] = {
            'available': False,
            'reason': 'Matching connectome geometry is unavailable.',
        }
        try:
            brain = resource(self.paths.brain, 'brain.npz')
            self.verify(path=brain, expected=manifest['brain_files_sha256']['brain.npz'])
            with np.load(file=brain, allow_pickle=False) as arrays:
                selected = np.flatnonzero(
                    a=(arrays['superclass'] == cfg.readout.population)
                    | (arrays['cell_type'] == cfg.readout.population)
                )
                positions = arrays['positions'][selected]
                if len(selected) != blocks[0].shape[1] or positions.shape != (len(selected), 3):
                    raise ValueError('Neuron geometry does not match the recorded features.')
                located = np.flatnonzero(a=np.isfinite(positions).all(axis=1))
                neurons = {
                    'available': True,
                    'feature_indices': located.tolist(),
                    'missing_positions': len(selected) - len(located),
                    'ids': arrays['ids'][selected].astype(str).tolist(),
                    'positions': positions[located].tolist(),
                    'population': cfg.readout.population,
                }
        except (OSError, ValueError, KeyError) as error:
            neurons['reason'] = str(error)
        builder = EpisodeBuilder(episodes=cfg.episodes, image_size=cfg.dataset.preprocess.size)
        return {
            'sample': sample,
            'image_size': cfg.dataset.preprocess.size,
            'boxes': builder.boxes_for(sample_id=sample_id).tolist(),
            'activity': dict(
                zip(cfg.readout.features, [block.tolist() for block in blocks], strict=True)
            ),
            'summary': np.asarray(a=store.array(name='summary')[sample['row']]).tolist(),
            'brain_ms': float(np.asarray(a=store.array(name='brain_ms')[sample['row']])),
            'neurons': neurons,
        }

    def image(self, run_id: str, sample_id: str) -> bytes:
        """Return an existing aligned RGB image without reading or altering source datasets.

        :param run_id: Run directory basename.
        :type run_id: str
        :param sample_id: Sample identity in the matching preprocessing cache.
        :type sample_id: str
        :returns: PNG bytes for the requested local image.
        :rtype: bytes
        :raises FileNotFoundError: If no matching prepared image exists.
        :raises ValueError: If recorded cache artifacts fail integrity checks.
        """
        manifest = load_manifest(run_dir=resource(self.paths.runs, run_id, 'manifest.json').parent)
        for source in self.paths.preprocess.glob('*/meta.json'):
            directory = resource(self.paths.preprocess, source.parent.name)
            meta = document(path=resource(directory, 'meta.json'))
            if meta.get('fingerprint') != manifest['dataset_fingerprint']:
                continue
            for name in ('images.npy', 'index.parquet'):
                self.verify(path=resource(directory, name), expected=meta['artifacts'][name])
            rows = read_table(path=resource(directory, 'index.parquet'))
            matches = [row for row in rows if row['sample_id'] == sample_id]
            if not matches:
                break
            images = np.load(
                file=resource(directory, 'images.npy'), mmap_mode='r', allow_pickle=False
            )
            pixels = images[matches[0]['row']]
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError('Prepared image has invalid RGB geometry.')
            stream = io.BytesIO()
            Image.fromarray(obj=pixels).save(fp=stream, format='PNG')
            return stream.getvalue()
        raise FileNotFoundError(
            'Aligned image cache is unavailable; metrics and traces remain viewable.'
        )

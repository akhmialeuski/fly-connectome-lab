"""Atomic, content-addressed aligned image caches with train-only templates."""

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from flystate import __version__
from flystate.datasets.align import align_face, canonical_template
from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.errors import DatasetError
from flystate.datasets.fingerprint import dataset_fingerprint, selected_images_hash
from flystate.datasets.registry import require_validated
from flystate.datasets.subset import Sample, select_samples
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj
from flystate.log import get_logger
from flystate.settings import Paths
from flystate.storage.json import write_json

CACHE_FORMAT: int = 1
CACHE_ARTIFACTS: tuple[str, ...] = ('images.npy', 'index.parquet', 'template.json')


@dataclass(frozen=True)
class PreparedDataset:
    """Ordered samples with read-only uint8 RGB images (N,S,S,3) and a (5,2) template."""

    key: str
    directory: Path
    samples: list[Sample]
    images: NDArray[np.uint8]
    template: NDArray[np.float64]
    fingerprint: str


def _load_cache(
    directory: Path, key: str, fingerprint: str, samples: list[Sample], size: int
) -> PreparedDataset:
    """Verify stored artifacts and expose a read-only memory-mapped image array.

    :param directory: Completed preprocessing cache.
    :type directory: Path
    :param key: Expected cache identity.
    :type key: str
    :param fingerprint: Expected semantic dataset identity.
    :type fingerprint: str
    :param samples: Expected ordered sample records.
    :type samples: list[Sample]
    :param size: Configured aligned image side.
    :type size: int
    :returns: Verified cache handle.
    :rtype: PreparedDataset
    :raises DatasetError: If metadata, content digests, schemas, or array shapes are invalid.
    """
    try:
        meta = json.loads(s=(directory / 'meta.json').read_text(encoding='utf-8'))
        if (
            meta['key'] != key
            or meta['fingerprint'] != fingerprint
            or meta['format'] != CACHE_FORMAT
        ):
            raise ValueError('Cache metadata identity mismatch.')
        for name in CACHE_ARTIFACTS:
            if sha256_file(path=directory / name) != meta['artifacts'][name]:
                raise ValueError(f'Cache artifact checksum mismatch: {name}.')
        index = pq.read_table(source=directory / 'index.parquet').to_pylist()
        expected = [{'row': row, **asdict(obj=sample)} for row, sample in enumerate(samples)]
        if index != expected:
            raise ValueError('Cache index disagrees with selected samples.')
        images = np.load(file=directory / 'images.npy', mmap_mode='r', allow_pickle=False)
        template = np.asarray(
            a=json.loads(s=(directory / 'template.json').read_text()), dtype=np.float64
        )
        if images.shape != (len(samples), size, size, 3) or images.dtype != np.uint8:
            raise ValueError('Cache image shape or dtype mismatch.')
        if template.shape != (5, 2) or not np.isfinite(template).all():
            raise ValueError('Cache template shape or values are invalid.')
        template.flags.writeable = False
        return PreparedDataset(
            key=key,
            directory=directory,
            samples=samples,
            images=images,
            template=template,
            fingerprint=fingerprint,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DatasetError(f'Invalid preprocessing cache {directory}: {error}') from error


def prepare_dataset(cfg: ExperimentConfig, paths: Paths) -> PreparedDataset:
    """Build or verify a cache whose identity includes selected source JPEG content.

    :param cfg: Validated experiment selection, split, and alignment parameters.
    :type cfg: ExperimentConfig
    :param paths: Registered source and cache storage locations.
    :type paths: Paths
    :returns: Ordered, aligned dataset backed by a read-only NPY memory map.
    :rtype: PreparedDataset
    :raises DatasetError: If sources are unvalidated, changed, or the cache is invalid.
    :raises OSError: If storage cannot hold the generated artifacts.
    """
    root = require_validated(paths=paths, name=cfg.dataset.name)
    adapter = CelebAAdapter(root=root, expected=None)
    records = adapter.records()
    samples = select_samples(records=records, subset=cfg.dataset.subset, split=cfg.dataset.split)
    annotation = adapter.annotation_fingerprint()
    image_hash = selected_images_hash(adapter=adapter, samples=samples)
    fingerprint = dataset_fingerprint(
        annotation=annotation,
        samples=samples,
        images_sha256=image_hash,
        preprocess=cfg.dataset.preprocess,
    )
    key = sha256_obj(
        obj={
            'format': CACHE_FORMAT,
            'annotation': annotation,
            'images_sha256': image_hash,
            'subset': cfg.dataset.subset.model_dump(mode='json'),
            'split': cfg.dataset.split.model_dump(mode='json'),
            'preprocess': cfg.dataset.preprocess.model_dump(mode='json'),
        }
    )
    directory = paths.preprocess / key
    size = cfg.dataset.preprocess.size
    if directory.exists():
        return _load_cache(
            directory=directory, key=key, fingerprint=fingerprint, samples=samples, size=size
        )
    paths.preprocess.mkdir(parents=True, exist_ok=True)
    temporary = paths.preprocess / f'{key}.tmp'
    try:
        temporary.mkdir()
    except FileExistsError as error:
        raise DatasetError(
            f'Preprocessing build already exists at {temporary}; '
            'check for an active writer or an interrupted build.'
        ) from error
    try:
        by_name = {record.filename: record for record in records}
        training = np.array(
            object=[by_name[item.filename].landmarks for item in samples if item.split == 'train'],
            dtype=np.float64,
        ).reshape(-1, 5, 2)
        template = canonical_template(train_landmarks=training, pre=cfg.dataset.preprocess)
        images = np.lib.format.open_memmap(
            filename=temporary / 'images.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(len(samples), size, size, 3),
        )
        try:
            for row, sample in enumerate(samples):
                landmarks = np.asarray(
                    a=by_name[sample.filename].landmarks, dtype=np.float64
                ).reshape(5, 2)
                images[row] = align_face(
                    image=adapter.load_image(filename=sample.filename),
                    landmarks=landmarks,
                    template=template,
                    size=size,
                )
            images.flush()
        finally:
            del images
        table = pa.Table.from_pylist(
            mapping=[{'row': row, **asdict(obj=sample)} for row, sample in enumerate(samples)]
        )
        pq.write_table(table=table, where=temporary / 'index.parquet')
        write_json(path=temporary / 'template.json', value=template.tolist())
        if selected_images_hash(adapter=adapter, samples=samples) != image_hash or (
            adapter.annotation_fingerprint() != annotation
        ):
            raise DatasetError('Dataset sources changed during preprocessing; validate and retry.')
        write_json(
            path=temporary / 'meta.json',
            value={
                'format': CACHE_FORMAT,
                'key': key,
                'fingerprint': fingerprint,
                'counts': dict(Counter(item.split for item in samples)),
                'created_utc': datetime.now(tz=UTC).isoformat(),
                'flystate_version': __version__,
                'artifacts': {name: sha256_file(path=temporary / name) for name in CACHE_ARTIFACTS},
            },
        )
        temporary.rename(target=directory)
    except BaseException:
        shutil.rmtree(path=temporary)
        raise
    get_logger(name='preprocess').info('dataset_prepared', cache_key=key, samples=len(samples))
    return _load_cache(
        directory=directory, key=key, fingerprint=fingerprint, samples=samples, size=size
    )

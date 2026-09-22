"""Cohort reservation, conservative duplicate screening, and effective graph provenance."""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import sparse

from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.preprocess import prepare_dataset
from flystate.datasets.registry import require_validated
from flystate.diagnostics.artifacts import attempt
from flystate.experiments.config import ExperimentConfig
from flystate.hashing import sha256_file, sha256_obj
from flystate.settings import Paths
from flystate.storage.json import write_json
from flystate.storage.parquet import read_table, write_table

RESERVE_PER_IDENTITY: int = 3
DHASH_DISTANCE: int = 3


def image_signatures(adapter: CelebAAdapter, filename: str) -> dict[str, Any]:
    """Compute file/RGB digests and a conservative 64-bit dHash without saving pixels.

    :param adapter: Validated local source adapter.
    :type adapter: CelebAAdapter
    :param filename: Safe annotated image basename.
    :type filename: str
    :returns: File and pixel digests, plus an integer perceptual hash.
    :rtype: dict[str, Any]
    """
    rgb = adapter.load_image(filename=filename)
    gray = np.asarray(
        a=Image.fromarray(obj=rgb)
        .convert(mode='L')
        .resize(size=(9, 8), resample=Image.Resampling.LANCZOS)
    )
    bits = np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes()
    pixel_digest = hashlib.sha256(string=str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
    return {
        'file_sha256': sha256_file(path=adapter.locate().images_dir / filename),
        'rgb_sha256': pixel_digest,
        'dhash': int.from_bytes(bytes=bits, byteorder='big'),
    }


def duplicate_kind(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    """Classify exact duplicates separately from perceptual review candidates.

    :param left: Source image signatures.
    :type left: dict[str, Any]
    :param right: Source image signatures.
    :type right: dict[str, Any]
    :returns: Exact, perceptual candidate, or null.
    :rtype: Optional[str]
    """
    if left['file_sha256'] == right['file_sha256'] or left['rgb_sha256'] == right['rgb_sha256']:
        return 'exact'
    if (left['dhash'] ^ right['dhash']).bit_count() <= DHASH_DISTANCE:
        return 'perceptual_candidate'
    return None


def graph_provenance(cfg: ExperimentConfig, paths: Paths) -> dict[str, Any]:
    """Hash the actual sensory-masked CSC arrays used by the CPU runtime.

    :param cfg: Source brain configuration.
    :type cfg: ExperimentConfig
    :param paths: Explicit brain-file directory layout.
    :type paths: Paths
    :returns: Source digests, effective edge count, and signed sparse-array identities.
    :rtype: dict[str, Any]
    """
    weights = sparse.load_npz(file=paths.brain / 'weights.npz')
    original_edges = weights.nnz
    with np.load(file=paths.brain / 'brain.npz', allow_pickle=False) as brain:
        sensory = np.char.find(brain['superclass'].astype(str), 'sensory') >= 0
    if not cfg.brain.sensory_input:
        weights = sparse.diags(diagonals=(~sensory).astype(np.float32)) @ weights.tocsr()
    effective = weights.tocsc()
    return {
        'source_files': {
            name: sha256_file(path=paths.brain / name) for name in ('brain.npz', 'weights.npz')
        },
        'sensory_input': cfg.brain.sensory_input,
        'sensory_neurons': int(sensory.sum()),
        'source_stored_edges': int(original_edges),
        'effective_stored_edges': int(effective.nnz),
        'positive_edges': int(np.count_nonzero(effective.data > 0)),
        'negative_edges': int(np.count_nonzero(effective.data < 0)),
        'csc_arrays': {
            name: {
                'dtype': str(array.dtype),
                'sha256': hashlib.sha256(string=array.tobytes()).hexdigest(),
            }
            for name, array in (
                ('indptr', effective.indptr),
                ('indices', effective.indices),
                ('data', effective.data),
            )
        },
    }


def audit_cohort(cfg: ExperimentConfig, paths: Paths, output: Path) -> dict[str, Any]:
    """Reserve unused photographs without examining model predictions or changing historical splits.

    :param cfg: Original development cohort configuration.
    :type cfg: ExperimentConfig
    :param paths: Source data and historical preprocessing caches.
    :type paths: Paths
    :param output: New attempt directory.
    :type output: Path
    :returns: Cohort coverage, duplicate candidates, reserve identity, and effective graph report.
    :rtype: dict[str, Any]
    """
    parameters = {
        'kind': 'cohort_audit',
        'reserve_per_identity': RESERVE_PER_IDENTITY,
        'perceptual_hash': 'grayscale horizontal 64-bit dHash, Pillow LANCZOS',
        'perceptual_hamming_threshold': DHASH_DISTANCE,
        'selection_seed': cfg.seed,
    }
    with attempt(paths=paths, cfg=cfg, output=output, parameters=parameters) as directory:
        prepared = prepare_dataset(cfg=cfg, paths=paths)
        adapter = CelebAAdapter(
            root=require_validated(paths=paths, name=cfg.dataset.name), expected=None
        )
        records = adapter.records()
        used: set[str] = set()
        historical_indexes = {}
        for index in sorted(paths.preprocess.glob('*/index.parquet')):
            used.update(row['filename'] for row in read_table(path=index))
            historical_indexes[index.parent.name] = sha256_file(path=index)
        selected = {sample.filename: sample for sample in prepared.samples}
        identities = sorted({sample.identity for sample in prepared.samples})
        by_name = {record.filename: record for record in records}
        signatures = {
            name: image_signatures(adapter=adapter, filename=name) for name in sorted(used)
        }
        duplicate_pairs = []
        names = sorted(selected)
        for offset, left in enumerate(names):
            for right in names[offset + 1 :]:
                if selected[left].split == selected[right].split:
                    continue
                kind = duplicate_kind(left=signatures[left], right=signatures[right])
                if kind is not None:
                    duplicate_pairs.append(
                        {
                            'left': left,
                            'right': right,
                            'left_split': selected[left].split,
                            'right_split': selected[right].split,
                            'kind': kind,
                        }
                    )
        candidates = [
            record
            for record in records
            if record.identity in identities and record.filename not in used
        ]
        candidates.sort(
            key=lambda record: sha256_obj(
                obj=[cfg.seed, record.identity, record.filename, 'confirmation-reserve']
            )
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        counts: Counter[int] = Counter()
        available = Counter(record.identity for record in candidates)
        for record in candidates:
            if counts[record.identity] >= RESERVE_PER_IDENTITY:
                continue
            signature = image_signatures(adapter=adapter, filename=record.filename)
            collision = next(
                (
                    (name, kind)
                    for name, other in signatures.items()
                    if (kind := duplicate_kind(left=signature, right=other)) is not None
                ),
                None,
            )
            if collision is not None:
                rejected.append(
                    {
                        'filename': record.filename,
                        'identity': record.identity,
                        'matches': collision[0],
                        'kind': collision[1],
                    }
                )
                continue
            accepted.append(
                {
                    'filename': record.filename,
                    'identity': record.identity,
                    'label': next(
                        s.label for s in prepared.samples if s.identity == record.identity
                    ),
                    **signature,
                }
            )
            signatures[record.filename] = signature
            counts[record.identity] += 1
        signature_rows = [
            {
                'filename': name,
                'identity': by_name[name].identity,
                'historical': name in used,
                **signature,
                'dhash': f'{signature["dhash"]:016x}',
            }
            for name, signature in sorted(signatures.items())
        ]
        write_table(path=directory / 'source-signatures.parquet', rows=signature_rows)
        reserve = [{**row, 'dhash': f'{row["dhash"]:016x}'} for row in accepted]
        write_json(path=directory / 'reserve.json', value=reserve)
        write_json(path=directory / 'duplicate-candidates.json', value=duplicate_pairs)
        write_json(path=directory / 'reserve-rejections.json', value=rejected)
        report = {
            'dataset_fingerprint': prepared.fingerprint,
            'annotation_fingerprint': adapter.annotation_fingerprint(),
            'historical_indexes': historical_indexes,
            'cohort_counts': dict(Counter(sample.split for sample in prepared.samples)),
            'reserve_coverage': [
                {
                    'identity': identity,
                    'available': available[identity],
                    'reserved': counts[identity],
                }
                for identity in identities
            ],
            'reserve_count': len(accepted),
            'reserve_sha256': sha256_obj(obj=reserve),
            'balanced_full_confirmation_available': all(
                counts[i] == RESERVE_PER_IDENTITY for i in identities
            ),
            'cross_split_duplicate_candidates': len(duplicate_pairs),
            'exact_duplicate_pairs': sum(pair['kind'] == 'exact' for pair in duplicate_pairs),
            'graph': graph_provenance(cfg=cfg, paths=paths),
            'decision': 'exploratory development; reserved images must not be fitted or scored',
            'perceptual_caveat': (
                'Candidates are conservatively excluded from reserve, not proven duplicates.'
            ),
        }
        write_json(path=directory / 'report.json', value=report)
    return report

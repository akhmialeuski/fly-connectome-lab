"""Explicit-location download, integrity verification, and connectome metadata."""

from pathlib import Path
from typing import TypedDict

import flybrain
import numpy as np
from flybrain import data
from scipy import sparse

from flystate.hashing import sha256_file
from flystate.settings import Paths

EXPECTED_NEURONS: int = 166_700
EXPECTED_CONNECTIONS: int = 25_582_938


class FileVerification(TypedDict):
    """File existence and optional comparison against the published digest."""

    exists: bool
    size_bytes: int
    sha256: str | None
    sha256_ok: bool | None


def download_brain(paths: Paths, force: bool = False) -> Path:
    """Fetch published brain files into the explicit experiment home.

    :param paths: Resolved experiment storage paths.
    :type paths: Paths
    :param force: Replace existing files when true.
    :type force: bool
    :returns: Brain directory.
    :rtype: Path
    :raises RuntimeError: If upstream download or checksum verification fails.
    """
    flybrain.download(data=paths.brain, force=force)
    return paths.brain


def verify_brain_files(brain_dir: Path, check_hash: bool = True) -> dict[str, FileVerification]:
    """Check every published brain file without loading arrays or downloading.

    :param brain_dir: Directory containing the two NPZ files.
    :type brain_dir: Path
    :param check_hash: Compute SHA-256 when true.
    :type check_hash: bool
    :returns: Per-file size in bytes and integrity status; absent hashes are null.
    :rtype: dict[str, FileVerification]
    :raises OSError: If an existing file cannot be read.
    """
    result: dict[str, FileVerification] = {}
    for name, expected in data.FILES.items():
        path = brain_dir / name
        exists = path.is_file()
        digest = sha256_file(path=path) if exists and check_hash else None
        result[name] = FileVerification(
            exists=exists,
            size_bytes=path.stat().st_size if exists else 0,
            sha256=digest,
            sha256_ok=digest == expected if digest is not None else None,
        )
    return result


def brain_stats(brain_dir: Path) -> dict[str, object]:
    """Read connectome dimensions and populations without constructing a simulator.

    :param brain_dir: Existing metadata and sparse weight directory.
    :type brain_dir: Path
    :returns: Neuron, synapse, photoreceptor, superclass and group counts.
    :rtype: dict[str, object]
    :raises ValueError: If metadata dimensions and the weight matrix disagree.
    :raises OSError: If required files cannot be read.
    """
    weights = sparse.load_npz(file=brain_dir / 'weights.npz')
    with np.load(file=brain_dir / 'brain.npz', allow_pickle=False) as meta:
        neurons = len(meta['ids'])
        if weights.shape != (neurons, neurons) or len(meta['superclass']) != neurons:
            raise ValueError('Brain metadata and weight matrix dimensions disagree.')
        names, counts = np.unique(ar=meta['superclass'], return_counts=True)
        return {
            'neurons': neurons,
            'connections': int(weights.nnz),
            'photoreceptors': len(meta['visual']),
            'superclass_counts': dict(zip(names.tolist(), counts.tolist(), strict=True)),
            'groups': sorted(
                key.removeprefix('group_') for key in meta.files if key.startswith('group_')
            ),
        }

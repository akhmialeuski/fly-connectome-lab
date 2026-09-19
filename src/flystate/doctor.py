"""Independent diagnostic checks for a local CPU experiment installation."""

import importlib
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import flybrain
import numba
import numpy as np
import psutil
from numba.core import config as numba_config
from numba.np.ufunc.parallel import threading_layer

from flystate.brain.files import verify_brain_files
from flystate.settings import Paths, ensure_dirs

REQUIRED_MODULES: tuple[str, ...] = (
    'numpy',
    'scipy',
    'numba',
    'flybrain',
    'typer',
    'pydantic',
    'yaml',
    'structlog',
    'psutil',
)
GIGABYTE: int = 10**9
MINIMUM_DISK_BYTES: int = 5 * GIGABYTE
RECOMMENDED_DISK_BYTES: int = 15 * GIGABYTE
RECOMMENDED_MEMORY_BYTES: int = 8 * GIGABYTE


class _NumbaConfiguration(Protocol):
    """Documented settings installed dynamically by Numba's environment loader."""

    NUMBA_NUM_THREADS: int


@dataclass(frozen=True)
class Check:
    """One ordered diagnostic result with an actionable explanation."""

    name: str
    status: Literal['ok', 'warn', 'error']
    detail: str


@numba.njit(parallel=True)
def _parallel_probe() -> int:
    """Initialize the Numba parallel runtime using a tiny reduction.

    :returns: Sum of probe indices.
    :rtype: int
    """
    return int(np.arange(2).sum())


def _imports() -> Check:
    """Try each required import and retain all failures.

    :returns: Combined import diagnostic.
    :rtype: Check
    """
    failures: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name=name)
        except (ImportError, OSError) as error:
            failures.append(f'{name}: {error}')
    return Check(
        name='imports',
        status='error' if failures else 'ok',
        detail='; '.join(failures) if failures else 'All required packages import.',
    )


def _home(paths: Paths) -> Check:
    """Create storage directories and verify actual write access.

    :param paths: Storage paths to inspect.
    :type paths: Paths
    :returns: Successful writable-home diagnostic.
    :rtype: Check
    :raises OSError: If directory creation or the write probe fails.
    """
    ensure_dirs(paths=paths)
    with tempfile.TemporaryFile(dir=paths.home) as probe:
        probe.write(b'flystate')
        probe.flush()
    return Check(name='home_writable', status='ok', detail=str(paths.home))


def _disk(paths: Paths) -> Check:
    """Compare home filesystem free space with experiment thresholds.

    :param paths: Storage paths to inspect.
    :type paths: Paths
    :returns: Available disk budget in decimal gigabytes.
    :rtype: Check
    """
    free = shutil.disk_usage(path=paths.home).free
    status: Literal['ok', 'warn', 'error'] = (
        'error' if free < MINIMUM_DISK_BYTES else 'warn' if free < RECOMMENDED_DISK_BYTES else 'ok'
    )
    return Check(name='disk_free', status=status, detail=f'{free / GIGABYTE:.2f} GB free')


def _threads() -> Check:
    """Initialize and describe the Numba CPU threading layer.

    :returns: Maximum/current thread counts and threading layer.
    :rtype: Check
    """
    _parallel_probe()
    # Numba installs this documented setting dynamically during configuration reload.
    maximum = cast(_NumbaConfiguration, numba_config).NUMBA_NUM_THREADS
    return Check(
        name='numba_threads',
        status='ok',
        detail=(f'maximum={maximum}, current={numba.get_num_threads()}, layer={threading_layer()}'),
    )


def _brain(paths: Paths, check_hash: bool) -> Check:
    """Distinguish absent connectomes from corrupt installed files.

    :param paths: Storage paths to inspect.
    :type paths: Paths
    :param check_hash: Verify published digests when true.
    :type check_hash: bool
    :returns: Connectome file diagnostic.
    :rtype: Check
    """
    files = verify_brain_files(brain_dir=paths.brain, check_hash=check_hash)
    missing = [name for name, value in files.items() if not value['exists']]
    corrupt = [name for name, value in files.items() if value['sha256_ok'] is False]
    if corrupt:
        return Check(name='brain_files', status='error', detail=f'Checksum mismatch: {corrupt}')
    if missing:
        return Check(name='brain_files', status='warn', detail=f'Missing: {missing}')
    return Check(
        name='brain_files',
        status='ok',
        detail='Published digests match.' if check_hash else 'Files present; hashes skipped.',
    )


def _datasets(paths: Paths) -> Check:
    """Report registered datasets without validating or changing them.

    :param paths: Storage paths to inspect.
    :type paths: Paths
    :returns: Dataset registry diagnostic.
    :rtype: Check
    :raises ValueError: If the registry root is not a JSON object.
    """
    source = paths.datasets / 'registry.json'
    registry = json.loads(s=source.read_text(encoding='utf-8')) if source.exists() else {}
    if not isinstance(registry, dict):
        raise ValueError('Dataset registry must be a JSON object.')
    if not registry:
        return Check(name='datasets', status='warn', detail='No datasets registered.')
    return Check(name='datasets', status='ok', detail=json.dumps(obj=registry, sort_keys=True))


def run_checks(paths: Paths, check_hash: bool = True) -> list[Check]:
    """Run all eleven diagnostics, preserving later checks after individual failures.

    :param paths: Resolved experiment directories.
    :type paths: Paths
    :param check_hash: Verify brain checksums when true.
    :type check_hash: bool
    :returns: Ordered diagnostics; errors do not prevent independent checks.
    :rtype: list[Check]
    """
    memory = psutil.virtual_memory().total
    wsl_mount = 'microsoft' in platform.release().lower() and str(paths.home).startswith('/mnt/')
    checks: list[tuple[str, Callable[[], Check]]] = [
        (
            'python',
            lambda: Check(
                name='python',
                status='ok' if sys.version_info[:2] == (3, 12) else 'error',
                detail=platform.python_version(),
            ),
        ),
        ('imports', _imports),
        ('home_writable', lambda: _home(paths=paths)),
        (
            'wsl_paths',
            lambda: Check(
                name='wsl_paths',
                status='warn' if wsl_mount else 'ok',
                detail='store FLYSTATE_HOME in the Linux file system'
                if wsl_mount
                else str(paths.home),
            ),
        ),
        ('disk_free', lambda: _disk(paths=paths)),
        (
            'memory',
            lambda: Check(
                name='memory',
                status='warn' if memory < RECOMMENDED_MEMORY_BYTES else 'ok',
                detail=f'{memory / GIGABYTE:.2f} GB total',
            ),
        ),
        (
            'cpu',
            lambda: Check(
                name='cpu',
                status='ok',
                detail=(
                    f'logical={psutil.cpu_count(logical=True)}, '
                    f'physical={psutil.cpu_count(logical=False)}'
                ),
            ),
        ),
        ('numba_threads', _threads),
        ('brain_files', lambda: _brain(paths=paths, check_hash=check_hash)),
        ('datasets', lambda: _datasets(paths=paths)),
        (
            'gpu',
            lambda: Check(
                name='gpu',
                status='ok',
                detail=f'available={flybrain.cuda_available()}; not used by flystate',
            ),
        ),
    ]
    results: list[Check] = []
    for name, check in checks:
        try:
            results.append(check())
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            results.append(Check(name=name, status='error', detail=str(error)))
    return results

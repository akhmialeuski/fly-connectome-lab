"""Runtime and source provenance for reproducible immutable experiment runs."""

import platform
import subprocess
from importlib.metadata import version
from typing import Any

import numba

from flystate.brain.benchmark import host_info

ENVIRONMENT_PACKAGES: tuple[str, ...] = (
    'numpy',
    'scipy',
    'numba',
    'scikit-learn',
    'zarr',
    'pyarrow',
)


def environment_report() -> dict[str, Any]:
    """Describe the active interpreter, numerical packages, CPU, and thread configuration.

    :returns: JSON-compatible environment provenance.
    :rtype: dict[str, Any]
    """
    return {
        **host_info(),
        'python': platform.python_version(),
        'numba_threads': numba.get_num_threads(),
        'versions': {
            package: version(distribution_name=package) for package in ENVIRONMENT_PACKAGES
        },
    }


def git_state() -> dict[str, str | bool | None]:
    """Inspect current source identity without modifying a checkout or assuming Git exists.

    :returns: Commit and dirty state, independently null when their lookup fails.
    :rtype: dict[str, str | bool | None]
    """
    try:
        commit = subprocess.run(
            args=['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': None, 'git_dirty': None}
    try:
        status = subprocess.run(
            args=['git', '--no-optional-locks', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {'git_commit': commit, 'git_dirty': bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': commit, 'git_dirty': None}

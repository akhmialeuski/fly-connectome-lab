"""Resolve the common experiment directory layout without import-time side effects."""

import os
from dataclasses import dataclass, fields
from pathlib import Path

HOME_ENVIRONMENT_VARIABLE: str = 'FLYSTATE_HOME'


@dataclass(frozen=True)
class Paths:
    """Absolute paths for one isolated experiment installation."""

    home: Path
    data: Path
    brain: Path
    datasets: Path
    cache: Path
    features: Path
    preprocess: Path
    runs: Path


def get_paths(home: Path | None = None) -> Paths:
    """Resolve explicit, environment, or working-directory storage.

    :param home: Optional explicit home directory.
    :type home: Optional[Path]
    :returns: Absolute paths, without creating directories.
    :rtype: Paths
    """
    root = (home or Path(os.environ.get(HOME_ENVIRONMENT_VARIABLE) or Path.cwd())).expanduser()
    root = root.resolve()
    return Paths(
        home=root,
        data=root / 'data',
        brain=root / 'data' / 'brain',
        datasets=root / 'data' / 'datasets',
        cache=root / 'cache',
        features=root / 'cache' / 'features',
        preprocess=root / 'cache' / 'preprocess',
        runs=root / 'runs',
    )


def ensure_dirs(paths: Paths) -> None:
    """Create the storage directories idempotently.

    :param paths: Resolved storage paths.
    :type paths: Paths
    :raises OSError: If storage is not writable.
    """
    for field in fields(class_or_instance=paths):
        getattr(paths, field.name).mkdir(parents=True, exist_ok=True)


def output_path(path: Path, paths: Paths) -> Path:
    """Resolve an export inside the experiment home.

    :param path: Requested absolute or home-relative output.
    :type path: Path
    :param paths: Storage boundary.
    :type paths: Paths
    :returns: Resolved output path.
    :rtype: Path
    :raises ValueError: If the path escapes the home or targets source directories.
    """
    resolved = (path if path.is_absolute() else paths.home / path).resolve()
    if not resolved.is_relative_to(paths.home):
        raise ValueError('Output must be inside FLYSTATE_HOME.')
    relative = resolved.relative_to(paths.home)
    if relative.parts and relative.parts[0] in {'src', 'configs', 'tests', '.git'}:
        raise ValueError('Output must not target source-controlled directories.')
    return resolved

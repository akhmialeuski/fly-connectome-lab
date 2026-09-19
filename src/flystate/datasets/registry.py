"""Atomic local dataset registration with content-bound validation status."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flystate.datasets.celeba import CelebAAdapter
from flystate.datasets.errors import DatasetError
from flystate.settings import Paths
from flystate.storage.json import write_json


def entries(paths: Paths) -> dict[str, Any]:
    """Read the registry without creating it.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :returns: Dataset-name to registration mapping.
    :rtype: dict[str, Any]
    :raises DatasetError: If the registry is unreadable or structurally invalid.
    """
    source = paths.datasets / 'registry.json'
    try:
        result = json.loads(s=source.read_text(encoding='utf-8')) if source.exists() else {}
        if not isinstance(result, dict) or not all(
            isinstance(value, dict) for value in result.values()
        ):
            raise ValueError('Registry entries must be JSON objects.')
        return result
    except (OSError, ValueError) as error:
        raise DatasetError(f'Cannot read dataset registry: {error}') from error


def register(paths: Paths, name: str, root: Path) -> dict[str, Any]:
    """Register an absolute local path and clear any previous validation.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :param name: Dataset name; only celeba is supported.
    :type name: str
    :param root: Existing dataset root directory.
    :type root: Path
    :returns: New registration entry.
    :rtype: dict[str, Any]
    :raises DatasetError: If the dataset kind or directory is unsupported.
    """
    if name != 'celeba':
        raise DatasetError('Only celeba is supported in POC 1.')
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise DatasetError(f'Dataset root is not a directory: {resolved}.')
    result = {
        'kind': name,
        'path': str(resolved),
        'registered_utc': datetime.now(tz=UTC).isoformat(),
        'validation': None,
    }
    data = entries(paths=paths)
    data[name] = result
    write_json(path=paths.datasets / 'registry.json', value=data)
    return result


def get(paths: Paths, name: str) -> dict[str, Any]:
    """Retrieve one registered dataset.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :param name: Registered dataset name.
    :type name: str
    :returns: Registration entry.
    :rtype: dict[str, Any]
    :raises DatasetError: If the dataset is not registered.
    """
    result = entries(paths=paths).get(name)
    if result is None:
        raise DatasetError(
            f'Dataset {name!r} is not registered; run flystate dataset register {name}.'
        )
    return result


def set_validation(paths: Paths, name: str, report: dict[str, object], full: bool) -> None:
    """Atomically retain the validation report and the exact annotation identity.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :param name: Registered dataset name.
    :type name: str
    :param report: Adapter validation report.
    :type report: dict[str, object]
    :param full: Whether image existence was verified.
    :type full: bool
    """
    record = get(paths=paths, name=name)
    record['validation'] = {**report, 'full': full, 'checked_utc': datetime.now(tz=UTC).isoformat()}
    data = entries(paths=paths)
    data[name] = record
    write_json(path=paths.datasets / 'registry.json', value=data)


def require_validated(paths: Paths, name: str) -> Path:
    """Require successful validation whose annotation fingerprint is still current.

    :param paths: Experiment storage paths.
    :type paths: Paths
    :param name: Registered dataset name.
    :type name: str
    :returns: Validated absolute dataset root.
    :rtype: Path
    :raises DatasetError: If registration, validation, or current annotation identity is missing.
    """
    hint = f'Run flystate dataset validate {name}.'
    try:
        record = get(paths=paths, name=name)
        root = Path(record['path'])
        validation = record.get('validation')
        if not isinstance(validation, dict) or validation.get('ok') is not True:
            raise DatasetError('Dataset has not passed validation.')
        fingerprint = CelebAAdapter(root=root, expected=None).annotation_fingerprint()
        if fingerprint != validation.get('annotation_fingerprint'):
            raise DatasetError('Dataset annotations changed after validation.')
        return root
    except (DatasetError, OSError, KeyError, TypeError) as error:
        raise DatasetError(f'{error} {hint}') from error

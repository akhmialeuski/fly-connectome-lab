"""Read-only artifact paths and finite JSON documents."""

import json
import re
from pathlib import Path
from typing import Any


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

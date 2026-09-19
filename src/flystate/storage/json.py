"""Atomic UTF-8 JSON metadata publication."""

import os
import tempfile
from pathlib import Path

from flystate.hashing import canonical_json


def write_json(path: Path, value: object) -> None:
    """Publish complete canonical JSON using a same-directory atomic replacement.

    :param path: Destination under the caller's experiment storage.
    :type path: Path
    :param value: JSON-serializable metadata, without nonfinite numbers.
    :type value: object
    :raises OSError: If writing or atomic replacement fails.
    :raises ValueError: If the metadata includes nonfinite numbers.
    """
    content = canonical_json(obj=value) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target=path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

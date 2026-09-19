"""Shared output and error contracts for command adapters."""

import sys

from flystate.hashing import canonical_json

OK: int = 0
RUNTIME_ERROR: int = 1
CONFIG_ERROR: int = 2
INCOMPATIBLE: int = 3
INTERRUPTED: int = 130


def emit(result: dict[str, object], as_json: bool) -> None:
    """Write a single result to standard output.

    :param result: JSON-compatible command result.
    :type result: dict[str, object]
    :param as_json: Whether to emit one canonical JSON object.
    :type as_json: bool
    """
    text = (
        canonical_json(obj=result)
        if as_json
        else '\n'.join(f'{key}: {value}' for key, value in result.items())
    )
    sys.stdout.write(text + '\n')

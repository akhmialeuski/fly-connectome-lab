"""Atomic typed Parquet publication for append-only evaluation artifacts."""

import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> None:
    """Publish a typed table atomically without replacing an existing artifact.

    :param path: New Parquet artifact path.
    :type path: Path
    :param rows: Column-value records.
    :type rows: list[dict[str, Any]]
    :param schema: Optional explicit physical column schema.
    :type schema: Optional[pa.Schema]
    :raises FileExistsError: If the artifact already exists.
    """
    if path.exists():
        raise FileExistsError(f'Artifact already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='wb', dir=path.parent, suffix='.parquet', delete_on_close=False
    ) as stream:
        pq.write_table(table=pa.Table.from_pylist(mapping=rows, schema=schema), where=stream)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        Path(stream.name).replace(target=path)


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet artifact as column-value records.

    :param path: Existing Parquet artifact.
    :type path: Path
    :returns: Records in stored order.
    :rtype: list[dict[str, Any]]
    """
    return pq.read_table(source=path).to_pylist()

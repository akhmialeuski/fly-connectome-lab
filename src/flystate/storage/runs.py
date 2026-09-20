"""Unique experiment directories with atomic writes and immutable completed artifacts."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import flybrain
import pyarrow as pa
import pyarrow.parquet as pq

from flystate import __version__
from flystate.experiments.config import ExperimentConfig, config_hash, effective_yaml
from flystate.settings import Paths
from flystate.storage.environment import environment_report, git_state
from flystate.storage.json import write_json


class RunError(ValueError):
    """A run is missing, malformed, immutable, or addressed outside its directory."""


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a run manifest and validate its minimum lifecycle contract.

    :param run_dir: Existing run directory.
    :type run_dir: Path
    :returns: Parsed manifest with recognized status and run identity.
    :rtype: dict[str, Any]
    :raises RunError: If the manifest is absent or malformed.
    """
    try:
        manifest = json.loads(s=(run_dir / 'manifest.json').read_text(encoding='utf-8'))
        if (
            manifest['schema_version'] != 1
            or manifest['run_id'] != run_dir.name
            or manifest['status'] not in {'running', 'failed', 'completed'}
        ):
            raise ValueError('Invalid manifest schema, identity, or status.')
        return manifest
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RunError(f'Cannot read run manifest {run_dir}: {error}') from error


def assert_mutable(run_dir: Path) -> None:
    """Reject writes to a run after its completed state is published.

    :param run_dir: Existing run directory.
    :type run_dir: Path
    :raises RunError: If the run is completed or its manifest is invalid.
    """
    if load_manifest(run_dir=run_dir)['status'] == 'completed':
        raise RunError('Completed run artifacts are immutable.')


def _target(run_dir: Path, relative: str) -> Path:
    """Validate mutability and resolve one artifact within its run.

    :param run_dir: Existing run directory.
    :type run_dir: Path
    :param relative: Relative artifact path.
    :type relative: str
    :returns: Absolute checked artifact path.
    :rtype: Path
    :raises RunError: If the run is immutable or the path escapes it.
    """
    assert_mutable(run_dir=run_dir)
    target = (run_dir / relative).resolve()
    if not target.is_relative_to(run_dir.resolve()) or Path(relative).is_absolute():
        raise RunError('Run artifacts must remain within their run directory.')
    return target


def write_run_json(run_dir: Path, relative: str, value: object) -> None:
    """Publish JSON under the run immutability guard.

    :param run_dir: Existing mutable run directory.
    :type run_dir: Path
    :param relative: Relative artifact path.
    :type relative: str
    :param value: JSON-compatible payload.
    :type value: object
    """
    write_json(path=_target(run_dir=run_dir, relative=relative), value=value)


def write_run_text(run_dir: Path, relative: str, value: str) -> None:
    """Atomically publish UTF-8 configuration or text under a mutable run.

    :param run_dir: Existing mutable run directory.
    :type run_dir: Path
    :param relative: Relative artifact path.
    :type relative: str
    :param value: Exact text to preserve.
    :type value: str
    """
    target = _target(run_dir=run_dir, relative=relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=target.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target=target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_run_parquet(run_dir: Path, relative: str, rows: list[dict[str, Any]]) -> None:
    """Atomically publish a tabular run artifact through the immutability guard.

    :param run_dir: Existing mutable run directory.
    :type run_dir: Path
    :param relative: Relative Parquet artifact path.
    :type relative: str
    :param rows: Schema-consistent column-value records.
    :type rows: list[dict[str, Any]]
    """
    target = _target(run_dir=run_dir, relative=relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, suffix='.parquet', delete=False
        ) as stream:
            temporary = Path(stream.name)
        pq.write_table(table=pa.Table.from_pylist(mapping=rows), where=temporary)
        temporary.replace(target=target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_manifest(run_dir: Path, **fields: Any) -> None:
    """Update mutable lifecycle fields while protecting immutable run identity.

    :param run_dir: Existing mutable run directory.
    :type run_dir: Path
    :param fields: New lifecycle or provenance fields.
    :type fields: Any
    :raises RunError: If the run is immutable or identity/status changes are invalid.
    """
    assert_mutable(run_dir=run_dir)
    if {'schema_version', 'run_id', 'created_utc'} & fields.keys():
        raise RunError('Run identity fields cannot be replaced.')
    manifest = {**load_manifest(run_dir=run_dir), **fields}
    if manifest['status'] not in {'running', 'failed', 'completed'}:
        raise RunError('Unknown run status.')
    write_json(path=run_dir / 'manifest.json', value=manifest)


def create_run(
    paths: Paths, cfg: ExperimentConfig, original_yaml: str, extra_manifest: dict[str, Any]
) -> Path:
    """Reserve a unique run ID and initialize source, environment, and lifecycle artifacts.

    :param paths: Experiment storage home.
    :type paths: Paths
    :param cfg: Effective validated experiment settings.
    :type cfg: ExperimentConfig
    :param original_yaml: Exact user-supplied configuration text.
    :type original_yaml: str
    :param extra_manifest: Cache, dataset, trajectory, thread, and brain provenance.
    :type extra_manifest: dict[str, Any]
    :returns: New mutable run directory.
    :rtype: Path
    """
    now = datetime.now(tz=UTC)
    stem = f'{now:%Y%m%d-%H%M%S}-{cfg.name}-{config_hash(cfg=cfg)[:6]}'
    paths.runs.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        run_dir = paths.runs / (stem if suffix == 1 else f'{stem}-{suffix}')
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
    manifest = {
        **extra_manifest,
        **git_state(),
        'schema_version': 1,
        'run_id': run_dir.name,
        'status': 'running',
        'created_utc': now.isoformat(),
        'finished_utc': None,
        'parent_run_id': None,
        'config_hash': config_hash(cfg=cfg),
        'seed': cfg.seed,
        'flystate_version': __version__,
        'flybrain_version': flybrain.__version__,
    }
    write_json(path=run_dir / 'manifest.json', value=manifest)
    try:
        write_run_text(run_dir=run_dir, relative='config.original.yaml', value=original_yaml)
        write_run_text(run_dir=run_dir, relative='config.yaml', value=effective_yaml(cfg=cfg))
        write_run_json(run_dir=run_dir, relative='environment.json', value=environment_report())
        write_run_text(run_dir=run_dir, relative='logs/events.jsonl', value='')
    except BaseException as error:
        update_manifest(
            run_dir=run_dir,
            status='failed',
            error=str(error),
            finished_utc=datetime.now(tz=UTC).isoformat(),
        )
        raise
    return run_dir


def resolve_run(paths: Paths, run: str) -> Path:
    """Resolve an explicit existing path or an ID within the experiment runs directory.

    :param paths: Experiment storage home.
    :type paths: Paths
    :param run: Existing path or run basename.
    :type run: str
    :returns: Validated absolute run directory.
    :rtype: Path
    :raises RunError: If no valid manifest exists at the resolved location.
    """
    candidate = Path(run).expanduser()
    directory = candidate.resolve() if candidate.is_dir() else (paths.runs / run).resolve()
    load_manifest(run_dir=directory)
    return directory

"""Trace simulation, integrity inspection, and chunked statistics commands."""

import logging
from pathlib import Path
from typing import Annotated, Literal

import typer

from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.datasets.errors import DatasetError
from flystate.experiments.config import ConfigError, load_config
from flystate.log import LOGGER_NAMESPACE, get_logger
from flystate.settings import get_paths
from flystate.traces.builder import build_trace, inspect_trace, open_trace, trace_stats

app = typer.Typer(no_args_is_help=True, help='Build, resume, and inspect neural feature caches.')


def _execute(
    config: Path,
    overrides: list[str] | None,
    as_json: bool,
    operation: Literal['build', 'inspect', 'stats'],
    force: bool = False,
    max_chunks: int | None = None,
) -> None:
    """Apply shared configuration, output, and failure handling to trace operations.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Optional dotted field overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit exactly one JSON report.
    :type as_json: bool
    :param operation: Requested trace operation.
    :type operation: Literal['build', 'inspect', 'stats']
    :param force: Rebuild the exact resolved cache under its writer lock.
    :type force: bool
    :param max_chunks: Optional trial-build chunk budget.
    :type max_chunks: Optional[int]
    :raises typer.Exit: If configuration, runtime, or interruption handling fails.
    """
    try:
        cfg = load_config(path=config, overrides=overrides or ())
        paths = get_paths()
        if operation == 'build':
            level = logging.getLogger(name=LOGGER_NAMESPACE).getEffectiveLevel()
            report = build_trace(
                cfg=cfg,
                paths=paths,
                force=force,
                max_chunks=max_chunks,
                verbosity=-1 if level >= logging.WARNING else int(level == logging.DEBUG),
            )
        else:
            store = open_trace(cfg=cfg, paths=paths, require_complete=operation == 'stats')
            report = (
                inspect_trace(store=store) if operation == 'inspect' else trace_stats(store=store)
            )
    except (
        ConfigError,
        DatasetError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        KeyboardInterrupt,
    ) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else 130
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='trace').error('trace_failed', detail=str(error), operation=operation)
        if as_json:
            emit(result={'error': str(error), 'interrupted': code == 130}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=report, as_json=as_json)


@app.command(name='build')
def build_command(
    config: Path,
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    force: Annotated[bool, typer.Option('--force')] = False,
    max_chunks: Annotated[int | None, typer.Option('--max-chunks', min=1)] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Simulate episodes into a verified cache, resuming after interruptions.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Optional dotted configuration overrides.
    :type overrides: Optional[list[str]]
    :param force: Delete and rebuild the resolved cache under exclusive writer control.
    :type force: bool
    :param max_chunks: Optional positive chunk budget for an interrupted trial build.
    :type max_chunks: Optional[int]
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    _execute(
        config=config,
        overrides=overrides,
        as_json=as_json,
        operation='build',
        force=force,
        max_chunks=max_chunks,
    )


@app.command(name='inspect')
def inspect_command(
    config: Path,
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Verify completed chunks and inspect cache dimensions and build status.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Optional dotted configuration overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    _execute(config=config, overrides=overrides, as_json=as_json, operation='inspect')


@app.command(name='stats')
def stats_command(
    config: Path,
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Summarize completed trace activity with bounded-memory chunk reads.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Optional dotted configuration overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    _execute(config=config, overrides=overrides, as_json=as_json, operation='stats')

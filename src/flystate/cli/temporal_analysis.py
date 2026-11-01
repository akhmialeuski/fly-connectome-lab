"""Analyze time-matched T27 response attempts without altering their files."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.temporal_analysis import analyze_temporal_pilot
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def temporal_analysis_command(
    config: Path,
    source: Annotated[Path, typer.Option('--source')],
    output: Annotated[Path, typer.Option('--output')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Analyze the nine fixed response attempts in a fresh immutable directory.

    :param config: Original effective experiment YAML.
    :type config: Path
    :param source: Existing parent directory containing all nine case attempts.
    :type source: Path
    :param output: Fresh analysis directory within FLYSTATE_HOME.
    :type output: Path
    :param as_json: Emit exactly one JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid arguments, source evidence, or runtime failure.
    """
    try:
        result = analyze_temporal_pilot(
            cfg=load_config(path=config),
            paths=get_paths(),
            source=source,
            output=output,
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='diagnose').error('temporal_analysis_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

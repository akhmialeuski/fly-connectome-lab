"""Analyze the frozen training-only identity-response conditions."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.identity_analysis import analyze_identity_access
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def identity_analysis_command(
    config: Path,
    source: Annotated[Path, typer.Option('--source')],
    output: Annotated[Path, typer.Option('--output')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    masks: Annotated[Path, typer.Option('--masks')],
    membership: Annotated[Path, typer.Option('--membership')],
    schedule: Annotated[Path, typer.Option('--schedule')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Verify three immutable response attempts and compute the fixed identity gate.

    :param config: Original effective experiment YAML.
    :type config: Path
    :param source: Parent of completed N0, N1, and OFF response attempts.
    :type source: Path
    :param output: Fresh analysis directory inside FLYSTATE_HOME.
    :type output: Path
    :param cohort: Frozen selected training-image document.
    :type cohort: Path
    :param masks: Frozen T27 anatomical observation-mask document.
    :type masks: Path
    :param membership: Archived original development membership JSON.
    :type membership: Path
    :param schedule: Frozen T28 machine-readable analysis schedule.
    :type schedule: Path
    :param as_json: Emit exactly one JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid source, runtime failure, or interruption.
    """
    try:
        result = analyze_identity_access(
            cfg=load_config(path=config),
            paths=get_paths(),
            source=source,
            output=output,
            cohort_path=cohort,
            masks_path=masks,
            membership_path=membership,
            schedule_path=schedule,
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='diagnose').error('identity_analysis_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

"""Run one frozen within-window fly response condition."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.temporal import CASES, run_temporal_case
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def temporal_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    case: Annotated[str, typer.Option('--case')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    masks: Annotated[Path, typer.Option('--masks')],
    membership: Annotated[Path, typer.Option('--membership')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Record one preregistered C0-C6 stimulus/blank condition without overwriting evidence.

    :param config: Original effective 20-identity experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory inside FLYSTATE_HOME.
    :type output: Path
    :param case: Frozen stimulus/blank condition name, C0-C6.
    :type case: str
    :param cohort: Frozen training-only cohort JSON document.
    :type cohort: Path
    :param masks: Frozen population-mask JSON document.
    :type masks: Path
    :param membership: Archived source train/validation membership JSON document.
    :type membership: Path
    :param as_json: Emit exactly one JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: On configuration, runtime, or interrupt failure.
    """
    try:
        if case not in CASES:
            raise ConfigError('Choose a frozen temporal case from C0 through C6.')
        cfg = load_config(path=config)
        result = run_temporal_case(
            cfg=cfg,
            paths=get_paths(),
            output=output,
            case_id=case,
            cohort_path=cohort,
            masks_path=masks,
            membership_path=membership,
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='diagnose').error('temporal_response_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

"""Run one frozen training-only identity-access response condition."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.identity import CASES, run_identity_case
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def identity_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    case: Annotated[str, typer.Option('--case')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    masks: Annotated[Path, typer.Option('--masks')],
    membership: Annotated[Path, typer.Option('--membership')],
    schedule: Annotated[Path, typer.Option('--schedule')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Record original-current identity responses without overwriting prior evidence.

    :param config: Original effective experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory inside FLYSTATE_HOME.
    :type output: Path
    :param case: Frozen N0, N1, or OFF condition.
    :type case: str
    :param cohort: Frozen two-training-photo-per-identity cohort JSON.
    :type cohort: Path
    :param masks: Frozen T27 anatomical observation-mask JSON.
    :type masks: Path
    :param membership: Archived original development membership JSON.
    :type membership: Path
    :param schedule: Frozen T28 machine-readable response schedule.
    :type schedule: Path
    :param as_json: Emit exactly one JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid configuration, runtime failure, or interruption.
    """
    try:
        if case not in CASES:
            raise ConfigError('Choose a frozen identity-access case from N0, N1, or OFF.')
        result = run_identity_case(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            case_id=case,
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
        get_logger(name='diagnose').error('identity_access_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

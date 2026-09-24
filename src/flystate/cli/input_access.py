"""Run the frozen training-only pixel/current identity-access baseline."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.input_access import run_input_access
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def input_access_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    schedule: Annotated[Path, typer.Option('--schedule')],
    membership: Annotated[Path, typer.Option('--membership')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Fit both frozen input probes on training photos and score training queries.

    :param config: Original experiment YAML with no overrides.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed 280-photograph source-hash and role document.
    :type cohort: Path
    :param schedule: Committed representation/CV/gate schedule.
    :type schedule: Path
    :param membership: Archived original train/validation/test membership document.
    :type membership: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = run_input_access(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            cohort_path=cohort,
            schedule_path=schedule,
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
        get_logger(name='diagnose').error('input_access_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

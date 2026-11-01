"""Run fit-only out-of-fold log-loss selection for input diagnostics."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.input_loss_selection import run_input_loss_selection
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def input_loss_selection_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    parent_schedule: Annotated[Path, typer.Option('--parent-schedule')],
    membership: Annotated[Path, typer.Option('--membership')],
    baseline: Annotated[Path, typer.Option('--baseline')],
    schedule: Annotated[Path, typer.Option('--schedule')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Choose input-probe C from fit-only OOF loss; report prior queries separately.

    :param config: Original effective experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory beneath FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed T29 exact cohort and source-hash file.
    :type cohort: Path
    :param parent_schedule: Committed T29 exact CV-fold file.
    :type parent_schedule: Path
    :param membership: Archived original split-membership file.
    :type membership: Path
    :param baseline: Archived completed T29 A1 attempt directory.
    :type baseline: Path
    :param schedule: Committed T30 selection/gate schedule.
    :type schedule: Path
    :param as_json: Emit exactly one result JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = run_input_loss_selection(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            membership_path=membership,
            baseline_path=baseline,
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
        get_logger(name='diagnose').error('input_loss_selection_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

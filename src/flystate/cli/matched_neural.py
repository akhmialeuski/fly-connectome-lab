"""Run the frozen matched input-versus-fly identity comparison."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.matched_neural import run_matched_neural
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def matched_neural_command(
    config: Path,
    output: Annotated[Path, typer.Option('--output')],
    cohort: Annotated[Path, typer.Option('--cohort')],
    parent_schedule: Annotated[Path, typer.Option('--parent-schedule')],
    input_schedule: Annotated[Path, typer.Option('--input-schedule')],
    input_attempt: Annotated[Path, typer.Option('--input-attempt')],
    membership: Annotated[Path, typer.Option('--membership')],
    schedule: Annotated[Path, typer.Option('--schedule')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Fit original fly traces on the same cohort and readout budget as inputs.

    :param config: Original persistent experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed T29 exact training cohort and source hashes.
    :type cohort: Path
    :param parent_schedule: Committed T29 exact fit folds.
    :type parent_schedule: Path
    :param input_schedule: Committed T30 input-selection schedule.
    :type input_schedule: Path
    :param input_attempt: Archived completed T30 input-control attempt.
    :type input_attempt: Path
    :param membership: Archived original dataset split membership.
    :type membership: Path
    :param schedule: Committed T31 trace/comparison/gate schedule.
    :type schedule: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = run_matched_neural(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            t30_schedule_path=input_schedule,
            t30_path=input_attempt,
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
        get_logger(name='diagnose').error('matched_neural_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

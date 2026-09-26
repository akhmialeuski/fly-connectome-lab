"""Record, evaluate and decide the T39 compound-eye identity ceiling."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import emit
from flystate.cli.drive_sweep import JSON_HELP, JSON_OPTION, OUTPUT_HELP, OUTPUT_OPTION, _fail
from flystate.diagnostics.confirmation import evaluate_confirmation
from flystate.diagnostics.eye import ISSUE, PIXEL_REFERENCE, decide, eye_cases, record_eye
from flystate.experiments.config import load_config
from flystate.settings import get_paths

RECORDING: str = 'eye'
BUDGET: int = 500000


def eye_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Sample every photograph through the modelled compound eye for every preregistered case.

    :param config: Cohort experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_eye(cfg=load_config(path=config), paths=get_paths(), output=output)
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='eye_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def eye_evaluate_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    recording: Annotated[Path, typer.Option('--recording', help='Completed eye recording.')],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Fit the standard readout on every eye case and the pixel reference, then score once.

    :param config: Cohort experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param recording: Completed ``eye-record`` attempt.
    :type recording: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = evaluate_confirmation(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            recordings={RECORDING: recording},
            populations=[PIXEL_REFERENCE, *eye_cases()],
            comparisons=[],
            issue=ISSUE,
            references=True,
            max_iterations=BUDGET,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='eye_evaluate_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def eye_decide_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    evaluation: Annotated[Path, typer.Option('--evaluation', help='Completed eye evaluation.')],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Apply the frozen T39 rule: retained fraction of above-chance accuracy at 90 degrees.

    :param config: Cohort experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param evaluation: Completed ``eye-evaluate`` attempt.
    :type evaluation: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = decide(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            evaluation=evaluation,
            recording_name=RECORDING,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='eye_decide_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

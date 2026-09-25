"""Record scaled-drive population responses and decode identity from them (T32)."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.drive_sweep import decode_drive, record_drive
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths

OUTPUT_OPTION: str = '--output'
COHORT_OPTION: str = '--cohort'
PARENT_OPTION: str = '--parent-schedule'
MEMBERSHIP_OPTION: str = '--membership'
JSON_OPTION: str = '--json'
OUTPUT_HELP: str = 'Fresh attempt directory.'
COHORT_HELP: str = 'Committed T29 280-image cohort document.'
PARENT_HELP: str = 'Committed T29 exact five-fold CV document.'
MEMBERSHIP_HELP: str = 'Archived original split membership document.'
JSON_HELP: str = 'Emit exactly one JSON object on stdout.'


def _fail(error: BaseException, event: str, as_json: bool) -> typer.Exit:
    """Log a failure, optionally emit its JSON form, and build the matching exit.

    :param error: Raised exception or interruption.
    :type error: BaseException
    :param event: Structured log event name.
    :type event: str
    :param as_json: Whether the caller requested JSON output.
    :type as_json: bool
    :returns: Exit carrying code 2 for configuration errors, 130 for interrupts, else 1.
    :rtype: typer.Exit
    """
    code = (
        CONFIG_ERROR
        if isinstance(error, ConfigError)
        else INTERRUPTED
        if isinstance(error, KeyboardInterrupt)
        else RUNTIME_ERROR
    )
    get_logger(name='diagnose').error(event, detail=str(error))
    if as_json:
        emit(result={'error': str(error)}, as_json=True)
    return typer.Exit(code=code)


def drive_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    cohort: Annotated[Path, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    parent_schedule: Annotated[Path, typer.Option(PARENT_OPTION, help=PARENT_HELP)],
    membership: Annotated[Path, typer.Option(MEMBERSHIP_OPTION, help=MEMBERSHIP_HELP)],
    amplitude_scale: Annotated[
        float, typer.Option('--amplitude-scale', help='Encoder current multiplier; 0 is blank.')
    ],
    noise: Annotated[bool, typer.Option('--noise/--no-noise', help='Episode noise.')] = True,
    episode_seed: Annotated[int, typer.Option('--episode-seed', help='Episode-noise seed.')] = 0,
    workers: Annotated[int, typer.Option('--workers', help='Simulator processes.')] = 3,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Record per-window spike counts of the T32 populations for 280 training photographs.

    :param config: Original persistent experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed T29 cohort document.
    :type cohort: Path
    :param parent_schedule: Committed T29 fold document.
    :type parent_schedule: Path
    :param membership: Archived split membership document.
    :type membership: Path
    :param amplitude_scale: Encoder current multiplier.
    :type amplitude_scale: float
    :param noise: Whether per-episode noise is applied.
    :type noise: bool
    :param episode_seed: Episode-noise seed.
    :type episode_seed: int
    :param workers: Number of simulator processes.
    :type workers: int
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_drive(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            membership_path=membership,
            amplitude_scale=amplitude_scale,
            noise_enabled=noise,
            episode_seed=episode_seed,
            workers=workers,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='drive_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def drive_decode_command(
    config: Path,
    recordings: Annotated[list[Path], typer.Argument(help='Completed drive-record attempts.')],
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    cohort: Annotated[Path, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    parent_schedule: Annotated[Path, typer.Option(PARENT_OPTION, help=PARENT_HELP)],
    membership: Annotated[Path, typer.Option(MEMBERSHIP_OPTION, help=MEMBERSHIP_HELP)],
    selection_schedule: Annotated[
        Path, typer.Option('--selection-schedule', help='Committed T30 readout schedule.')
    ],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Decode identity from recorded conditions with the T30 fit-only OOF procedure.

    :param config: Original persistent experiment YAML.
    :type config: Path
    :param recordings: Completed drive-record attempt directories.
    :type recordings: list[Path]
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed T29 cohort document.
    :type cohort: Path
    :param parent_schedule: Committed T29 fold document.
    :type parent_schedule: Path
    :param membership: Archived split membership document.
    :type membership: Path
    :param selection_schedule: Committed T30 C grid, PCA and convergence schedule.
    :type selection_schedule: Path
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = decode_drive(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            recordings=recordings,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            membership_path=membership,
            selection_schedule_path=selection_schedule,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='drive_decode_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

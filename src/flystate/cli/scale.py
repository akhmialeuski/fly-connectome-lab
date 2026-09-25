"""Record, evaluate and analyze the T38 scale, encoder-seed and forgetting study."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import emit
from flystate.cli.drive_sweep import JSON_HELP, JSON_OPTION, OUTPUT_HELP, OUTPUT_OPTION, _fail
from flystate.cli.wiring import _pairs
from flystate.diagnostics.confirmation import DELAY_INFIX, evaluate_confirmation
from flystate.diagnostics.scale import (
    INTERFERENCE,
    ISSUE,
    KEEP,
    PERSISTENT,
    PRIMARY,
    RESET,
    analyze_scale,
    record_scale,
)
from flystate.experiments.config import load_config
from flystate.settings import get_paths

SET_OPTION: str = '--set'
SET_HELP: str = 'Dotted configuration override, for example encoder.seed=1.'
DELAY_OPTION: str = '--delay'
DELAY_HELP: str = 'Blank windows after the last glimpse at which states are kept (repeatable).'
CASE_SEPARATOR: str = '/'


def scale_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    gain: Annotated[float, typer.Option('--gain', help='Multiplier of the effective weights.')],
    leak: Annotated[float, typer.Option('--leak', help='Leak of the non-driven neurons.')],
    driven_leak: Annotated[float, typer.Option('--driven-leak', help='Leak of driven neurons.')],
    input_scale: Annotated[
        float, typer.Option('--input-scale', help='Multiplier of the encoded current.')
    ],
    steps_per_window: Annotated[
        int, typer.Option('--steps-per-window', help='Synaptic updates per window.')
    ],
    batch_size: Annotated[
        int, typer.Option('--batch-size', help='Photographs simulated together.')
    ],
    delays: Annotated[list[int] | None, typer.Option(DELAY_OPTION, help=DELAY_HELP)] = None,
    reset_each_window: Annotated[
        bool,
        typer.Option('--reset-each-window/--persistent', help='Zero the state before each window.'),
    ] = False,
    interference: Annotated[
        bool,
        typer.Option(
            '--interference/--blank',
            help="After the last glimpse show another identity's glimpses instead of blanks.",
        ),
    ] = False,
    overrides: Annotated[list[str] | None, typer.Option(SET_OPTION, help=SET_HELP)] = None,
    threads: Annotated[
        int | None, typer.Option('--threads', help='Numba threads; default from the config.')
    ] = None,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Record a cohort's final states in photograph batches, optionally after blank windows.

    :param config: Experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: Leak of the non-driven neurons.
    :type leak: float
    :param driven_leak: Leak of the driven neurons.
    :type driven_leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param batch_size: Photographs simulated together; must divide the cohort.
    :type batch_size: int
    :param delays: Blank delays; delay 0 only when omitted.
    :type delays: Optional[list[int]]
    :param reset_each_window: Whether to zero the state before each window.
    :type reset_each_window: bool
    :param interference: Whether windows after the last glimpse show a partner photograph.
    :type interference: bool
    :param overrides: Dotted configuration overrides.
    :type overrides: Optional[list[str]]
    :param threads: Numba threads.
    :type threads: Optional[int]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_scale(
            cfg=load_config(path=config, overrides=overrides or ()),
            paths=get_paths(),
            output=output,
            gain=gain,
            leak=leak,
            driven_leak=driven_leak,
            input_scale=input_scale,
            steps_per_window=steps_per_window,
            reset_each_window=reset_each_window,
            delays=tuple(delays or (0,)),
            batch_size=batch_size,
            threads=threads,
            interference=interference,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='scale_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def scale_evaluate_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    persistent: Annotated[
        Path, typer.Option('--persistent-recording', help='Persistent recording with delays.')
    ],
    reset: Annotated[Path, typer.Option('--reset-recording', help='Reset recording.')],
    interference: Annotated[
        Path | None,
        typer.Option('--interference-recording', help='Optional interference recording.'),
    ] = None,
    delays: Annotated[list[int] | None, typer.Option(DELAY_OPTION, help=DELAY_HELP)] = None,
    references: Annotated[
        bool,
        typer.Option('--references/--no-references', help='Also score the input references.'),
    ] = False,
    overrides: Annotated[list[str] | None, typer.Option(SET_OPTION, help=SET_HELP)] = None,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Fit the standard readout on training photographs and score every held-out photograph once.

    :param config: Experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param persistent: Completed persistent recording.
    :type persistent: Path
    :param reset: Completed reset recording.
    :type reset: Path
    :param interference: Completed interference recording with the same delays, or none.
    :type interference: Optional[Path]
    :param delays: Blank delays recorded in the persistent recording.
    :type delays: Optional[list[int]]
    :param references: Whether the three input-reference cases are scored.
    :type references: bool
    :param overrides: Dotted configuration overrides identical to the recordings'.
    :type overrides: Optional[list[str]]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        delayed = [f'{PRIMARY}{DELAY_INFIX}{delay}' for delay in delays or () if delay > 0]
        recordings = {PERSISTENT: persistent, RESET: reset}
        only = {RESET: list(KEEP)}
        if interference is not None:
            recordings[INTERFERENCE] = interference
            only[INTERFERENCE] = delayed
        result = evaluate_confirmation(
            cfg=load_config(path=config, overrides=overrides or ()),
            paths=get_paths(),
            output=output,
            recordings=recordings,
            populations=[*KEEP, *delayed],
            comparisons=[
                *(
                    (f'{PERSISTENT}{CASE_SEPARATOR}{name}', f'{RESET}{CASE_SEPARATOR}{name}')
                    for name in KEEP
                ),
                *(
                    (
                        f'{source}{CASE_SEPARATOR}{delayed[-1]}',
                        f'{PERSISTENT}{CASE_SEPARATOR}{PRIMARY}',
                    )
                    for source in (PERSISTENT, *([INTERFERENCE] if interference else []))
                    if delayed
                ),
            ],
            issue=ISSUE,
            references=references,
            recording_populations=only,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='scale_evaluate_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def scale_analyze_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    development: Annotated[
        list[str], typer.Option('--development', help='Development evaluation as seed=directory.')
    ],
    confirmation: Annotated[
        Path, typer.Option('--confirmation', help='Confirmation evaluation directory.')
    ],
    delays: Annotated[list[int], typer.Option(DELAY_OPTION, help=DELAY_HELP)],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Apply the frozen T38 decision rules to every development seed and the confirmation.

    :param config: Development experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param development: Encoder-seed label and evaluation directory pairs.
    :type development: list[str]
    :param confirmation: Completed confirmation evaluation.
    :type confirmation: Path
    :param delays: Blank delays, starting with 0.
    :type delays: list[int]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        result = analyze_scale(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            development={name: Path(path) for name, path in _pairs(items=development)},
            confirmation=confirmation,
            delays=delays,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='scale_analyze_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

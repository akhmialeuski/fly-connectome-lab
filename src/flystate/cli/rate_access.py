"""Record MaleCNS population states under deterministic graded dynamics (T33)."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import emit
from flystate.cli.drive_sweep import (
    COHORT_HELP,
    COHORT_OPTION,
    JSON_HELP,
    JSON_OPTION,
    MEMBERSHIP_HELP,
    MEMBERSHIP_OPTION,
    OUTPUT_HELP,
    OUTPUT_OPTION,
    PARENT_HELP,
    PARENT_OPTION,
    _fail,
)
from flystate.diagnostics.rate_access import analyze_memory, memory_curves, record_rate
from flystate.experiments.config import load_config
from flystate.settings import get_paths


def rate_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    cohort: Annotated[Path, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    parent_schedule: Annotated[Path, typer.Option(PARENT_OPTION, help=PARENT_HELP)],
    membership: Annotated[Path, typer.Option(MEMBERSHIP_OPTION, help=MEMBERSHIP_HELP)],
    gain: Annotated[float, typer.Option('--gain', help='Multiplier of the effective weights.')],
    leak: Annotated[float, typer.Option('--leak', help='State update fraction in (0, 1].')],
    input_scale: Annotated[
        float, typer.Option('--input-scale', help='Multiplier of the encoded current.')
    ],
    steps_per_window: Annotated[
        int, typer.Option('--steps-per-window', help='Synaptic updates per window.')
    ],
    driven_leak: Annotated[
        float | None,
        typer.Option('--driven-leak', help='Separate leak of the driven neurons; default common.'),
    ] = None,
    reset_each_window: Annotated[
        bool,
        typer.Option('--reset-each-window/--persistent', help='Zero the state before each window.'),
    ] = False,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Record window-end states of the T32 populations under leaky-tanh dynamics.

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
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: State update fraction.
    :type leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param driven_leak: Separate leak of the driven neurons, or the common leak.
    :type driven_leak: Optional[float]
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param reset_each_window: Whether to zero the state before each window.
    :type reset_each_window: bool
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_rate(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            membership_path=membership,
            gain=gain,
            leak=leak,
            driven_leak=driven_leak,
            input_scale=input_scale,
            steps_per_window=steps_per_window,
            reset_each_window=reset_each_window,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='rate_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def rate_memory_command(
    config: Path,
    decodes: Annotated[list[Path], typer.Argument(help='Completed drive-decode attempts.')],
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    settings: Annotated[
        list[str], typer.Option('--setting', help='Setting with -persistent and -reset runs.')
    ],
    populations: Annotated[list[str], typer.Option('--population', help='Population to test.')],
    representation: Annotated[
        str, typer.Option('--representation', help='Decoded representation to compare.')
    ] = 'last_window_state',
    minimum_gain_pp: Annotated[
        float, typer.Option('--minimum-gain-pp', help='Required persistent gain, points.')
    ] = 10.0,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Test memory held by the network: persistent versus reset states, paired by photograph.

    :param config: Original persistent experiment YAML.
    :type config: Path
    :param decodes: Completed decode attempts with every persistent and reset condition.
    :type decodes: list[Path]
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param settings: Dynamics setting names.
    :type settings: list[str]
    :param populations: Populations to test.
    :type populations: list[str]
    :param representation: Decoded representation to compare.
    :type representation: str
    :param minimum_gain_pp: Required persistent-minus-reset accuracy, percentage points.
    :type minimum_gain_pp: float
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = analyze_memory(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            decodes=decodes,
            settings=settings,
            populations=populations,
            representation=representation,
            minimum_gain_pp=minimum_gain_pp,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='rate_memory_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def rate_memory_curve_command(
    config: Path,
    recordings: Annotated[list[Path], typer.Argument(help='Completed rate-record attempts.')],
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    cohort: Annotated[Path, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    parent_schedule: Annotated[Path, typer.Option(PARENT_OPTION, help=PARENT_HELP)],
    membership: Annotated[Path, typer.Option(MEMBERSHIP_OPTION, help=MEMBERSHIP_HELP)],
    populations: Annotated[list[str], typer.Option('--population', help='Population to test.')],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Measure, without labels, how well final states recall each window's input.

    :param config: Original persistent experiment YAML.
    :type config: Path
    :param recordings: Completed rate-record attempt directories.
    :type recordings: list[Path]
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param cohort: Committed T29 cohort document.
    :type cohort: Path
    :param parent_schedule: Committed T29 fold document.
    :type parent_schedule: Path
    :param membership: Archived split membership document.
    :type membership: Path
    :param populations: Populations whose final state is analysed.
    :type populations: list[str]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = memory_curves(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            recordings=recordings,
            cohort_path=cohort,
            parent_schedule_path=parent_schedule,
            membership_path=membership,
            populations=populations,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='rate_memory_curve_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

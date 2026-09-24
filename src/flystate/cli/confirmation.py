"""Record and evaluate the single frozen T35 confirmation on untouched identities."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import emit
from flystate.cli.drive_sweep import JSON_HELP, JSON_OPTION, OUTPUT_HELP, OUTPUT_OPTION, _fail
from flystate.diagnostics.confirmation import evaluate_confirmation, record_confirmation
from flystate.experiments.config import ConfigError, load_config
from flystate.settings import get_paths

PAIR_SEPARATOR: str = '='
COMPARISON_SEPARATOR: str = ':'


def confirm_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
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
    shuffle_seed: Annotated[
        int | None,
        typer.Option('--shuffle-seed', help='Degree-preserving shuffle seed; default fly graph.'),
    ] = None,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Record final graded states for every photograph of the confirmation cohort.

    :param config: Confirmation experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param gain: Multiplier of the row-normalized effective weights.
    :type gain: float
    :param leak: State update fraction.
    :type leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param driven_leak: Separate leak of the driven neurons, or the common leak.
    :type driven_leak: Optional[float]
    :param reset_each_window: Whether to zero the state before each window.
    :type reset_each_window: bool
    :param shuffle_seed: Degree-preserving shuffle seed, or the unchanged fly graph.
    :type shuffle_seed: Optional[int]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_confirmation(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            gain=gain,
            leak=leak,
            driven_leak=driven_leak,
            input_scale=input_scale,
            steps_per_window=steps_per_window,
            reset_each_window=reset_each_window,
            shuffle_seed=shuffle_seed,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='confirm_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def confirm_evaluate_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    recordings: Annotated[
        list[str], typer.Option('--recording', help='Recording as name=attempt-directory.')
    ],
    populations: Annotated[list[str], typer.Option('--population', help='Population to score.')],
    comparisons: Annotated[
        list[str] | None,
        typer.Option('--compare', help='Paired comparison as caseA:caseB (A minus B).'),
    ] = None,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Fit on training photographs and score every untouched held-out photograph once.

    :param config: Confirmation experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param recordings: Recording name and attempt directory pairs.
    :type recordings: list[str]
    :param populations: Populations whose final state is scored.
    :type populations: list[str]
    :param comparisons: Paired comparisons of case names.
    :type comparisons: Optional[list[str]]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        if any(PAIR_SEPARATOR not in item for item in recordings) or any(
            COMPARISON_SEPARATOR not in item for item in comparisons or []
        ):
            raise ConfigError('Use name=directory for recordings and caseA:caseB for comparisons.')
        pairs = [item.split(PAIR_SEPARATOR, 1) for item in recordings]
        result = evaluate_confirmation(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            recordings={name: Path(path) for name, path in pairs},
            populations=populations,
            comparisons=[
                (first, second)
                for first, second in (
                    item.rsplit(COMPARISON_SEPARATOR, 1) for item in comparisons or []
                )
            ],
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='confirm_evaluate_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

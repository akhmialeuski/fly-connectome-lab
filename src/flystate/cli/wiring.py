"""Record, evaluate, select and analyze the T37 wiring-specificity study."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import emit
from flystate.cli.drive_sweep import JSON_HELP, JSON_OPTION, OUTPUT_HELP, OUTPUT_OPTION, _fail
from flystate.diagnostics.artifacts import export_linear_readout
from flystate.diagnostics.confirmation import evaluate_confirmation
from flystate.diagnostics.wiring import (
    ISSUE,
    KEEP,
    Contrast,
    analyze_wiring,
    record_wiring,
    select_operating_points,
)
from flystate.experiments.config import ConfigError, ExperimentConfig, load_config
from flystate.settings import get_paths

PAIR_SEPARATOR: str = '='
CASE_SEPARATOR: str = ':'
LIST_SEPARATOR: str = ','
COHORT_PREFIX: str = 's'
COHORT_OPTION: str = '--cohort'
EVALUATION_OPTION: str = '--evaluation'
COHORT_HELP: str = 'Identity-selection seed of a cohort; the cohort is named s<seed>.'
PAIR_ERROR: str = 'Expected name=value, received {item!r}.'


def cohort(config: Path, seed: int) -> tuple[str, ExperimentConfig]:
    """Load the base configuration with another identity-selection seed.

    :param config: Base experiment YAML, identical for every cohort.
    :type config: Path
    :param seed: Identity-selection seed of the cohort.
    :type seed: int
    :returns: Cohort name ``s<seed>`` and its configuration, named ``<base>-s<seed>``.
    :rtype: tuple[str, ExperimentConfig]
    """
    name = f'{COHORT_PREFIX}{seed}'
    base = load_config(path=config)
    return name, load_config(
        path=config,
        overrides=[f'dataset.subset.selection_seed={seed}', f'name={base.name}-{name}'],
    )


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    """Split ``name=value`` command-line items.

    :param items: Raw items.
    :type items: list[str]
    :returns: Name and value pairs in the given order.
    :rtype: list[tuple[str, str]]
    :raises ConfigError: If an item has no separator.
    """
    if any(PAIR_SEPARATOR not in item for item in items):
        raise ConfigError(PAIR_ERROR.format(item=items))
    return [(name, value) for name, value in (item.split(PAIR_SEPARATOR, 1) for item in items)]


def wiring_record_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    seeds: Annotated[list[int], typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    family: Annotated[
        str, typer.Option('--family', help='fly, degree, random_target or feedforward.')
    ],
    leak: Annotated[float, typer.Option('--leak', help='Leak of the non-driven neurons.')],
    driven_leak: Annotated[float, typer.Option('--driven-leak', help='Leak of driven neurons.')],
    input_scale: Annotated[
        float, typer.Option('--input-scale', help='Multiplier of the encoded current.')
    ],
    steps_per_window: Annotated[
        int, typer.Option('--steps-per-window', help='Synaptic updates per window.')
    ],
    graph_seed: Annotated[
        int | None, typer.Option('--graph-seed', help='Null-model seed of seeded families.')
    ] = None,
    alpha: Annotated[
        float | None,
        typer.Option('--alpha', help='Gain as alpha over the giant-component spectral radius.'),
    ] = None,
    gain: Annotated[float | None, typer.Option('--gain', help='Explicit gain.')] = None,
    reset_each_window: Annotated[
        bool,
        typer.Option('--reset-each-window/--persistent', help='Zero the state before each window.'),
    ] = False,
    threads: Annotated[
        int | None, typer.Option('--threads', help='Numba threads; default from the config.')
    ] = None,
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Build one graph and record the final states of every listed cohort on it.

    :param config: Base experiment YAML.
    :type config: Path
    :param output: Directory within FLYSTATE_HOME that receives one attempt per cohort.
    :type output: Path
    :param seeds: Identity-selection seeds of the cohorts.
    :type seeds: list[int]
    :param family: Graph family.
    :type family: str
    :param leak: Leak of the non-driven neurons.
    :type leak: float
    :param driven_leak: Leak of the driven neurons.
    :type driven_leak: float
    :param input_scale: Multiplier of the encoded current.
    :type input_scale: float
    :param steps_per_window: Synaptic updates per window.
    :type steps_per_window: int
    :param graph_seed: Null-model seed.
    :type graph_seed: Optional[int]
    :param alpha: Operating point relative to the spectral radius.
    :type alpha: Optional[float]
    :param gain: Explicit gain.
    :type gain: Optional[float]
    :param reset_each_window: Whether to zero the state before each window.
    :type reset_each_window: bool
    :param threads: Numba threads.
    :type threads: Optional[int]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config, runtime failure, or interruption.
    """
    try:
        result = record_wiring(
            cohorts=dict(cohort(config=config, seed=seed) for seed in seeds),
            paths=get_paths(),
            output=output,
            family=family,
            seed=graph_seed,
            alpha=alpha,
            gain=gain,
            leak=leak,
            driven_leak=driven_leak,
            input_scale=input_scale,
            steps_per_window=steps_per_window,
            reset_each_window=reset_each_window,
            threads=threads,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='wiring_record_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def wiring_evaluate_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    seed: Annotated[int, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    recordings: Annotated[
        list[str], typer.Option('--recording', help='Recording as name=attempt-directory.')
    ],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Fit the standard readout of every recording of one cohort and score its held-out photos.

    :param config: Base experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param seed: Identity-selection seed of the cohort.
    :type seed: int
    :param recordings: Recording name and attempt directory pairs.
    :type recordings: list[str]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        result = evaluate_confirmation(
            cfg=cohort(config=config, seed=seed)[1],
            paths=get_paths(),
            output=output,
            recordings={name: Path(path) for name, path in _pairs(items=recordings)},
            populations=list(KEEP),
            comparisons=[],
            issue=ISSUE,
            references=False,
            export=export_linear_readout,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='wiring_evaluate_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def wiring_select_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    seed: Annotated[int, typer.Option(COHORT_OPTION, help=COHORT_HELP)],
    evaluation: Annotated[Path, typer.Option(EVALUATION_OPTION, help='Development evaluation.')],
    candidates: Annotated[
        list[str], typer.Option('--candidate', help='Candidate as family:alpha=case.')
    ],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Freeze each family's alpha by training-only CV accuracy on the development cohort.

    :param config: Base experiment YAML.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param seed: Identity-selection seed of the development cohort.
    :type seed: int
    :param evaluation: Completed development evaluation attempt.
    :type evaluation: Path
    :param candidates: ``family:alpha=case`` items.
    :type candidates: list[str]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        table: dict[str, dict[float, str]] = {}
        for key, case in _pairs(items=candidates):
            family, _, alpha = key.partition(CASE_SEPARATOR)
            table.setdefault(family, {})[float(alpha)] = case
        result = select_operating_points(
            cfg=cohort(config=config, seed=seed)[1],
            paths=get_paths(),
            evaluation=evaluation,
            output=output,
            candidates=table,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='wiring_select_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)


def wiring_analyze_command(
    config: Path,
    output: Annotated[Path, typer.Option(OUTPUT_OPTION, help=OUTPUT_HELP)],
    evaluations: Annotated[
        list[str], typer.Option(EVALUATION_OPTION, help='Cohort evaluation as name=directory.')
    ],
    contrasts: Annotated[
        list[str],
        typer.Option('--contrast', help='Contrast as name=case:ensemble-case,ensemble-case.'),
    ],
    as_json: Annotated[bool, typer.Option(JSON_OPTION, help=JSON_HELP)] = False,
) -> None:
    """Pool every cohort and estimate each preregistered contrast with its decision.

    :param config: Base experiment YAML; its seed drives the bootstrap.
    :type config: Path
    :param output: Fresh attempt directory within FLYSTATE_HOME.
    :type output: Path
    :param evaluations: Cohort name and evaluation directory pairs.
    :type evaluations: list[str]
    :param contrasts: ``name=case:ensemble,...`` items.
    :type contrasts: list[str]
    :param as_json: Emit exactly one result JSON object to stdout.
    :type as_json: bool
    :raises typer.Exit: On invalid config or arguments, runtime failure, or interruption.
    """
    try:
        parsed = []
        for name, value in _pairs(items=contrasts):
            first, separator, ensemble = value.partition(CASE_SEPARATOR)
            if not separator or not ensemble:
                raise ConfigError(f'Expected case:ensemble for contrast {name}.')
            parsed.append(
                Contrast(name=name, first=first, ensemble=tuple(ensemble.split(LIST_SEPARATOR)))
            )
        result = analyze_wiring(
            cfg=load_config(path=config),
            paths=get_paths(),
            output=output,
            evaluations={name: Path(path) for name, path in _pairs(items=evaluations)},
            contrasts=parsed,
        )
    except (Exception, KeyboardInterrupt) as error:
        raise _fail(error=error, event='wiring_analyze_failed', as_json=as_json) from error
    emit(result=result, as_json=as_json)

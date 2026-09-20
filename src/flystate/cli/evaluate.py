"""Evaluate immutable training runs with append-only prediction artifacts."""

from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.evaluation.evaluate import evaluate
from flystate.experiments.config import ConfigError
from flystate.log import get_logger
from flystate.settings import get_paths
from flystate.storage.runs import resolve_run


def evaluate_command(
    run: str,
    split: Annotated[str, typer.Option('--split')] = 'test',
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Report per-observation accuracy and predictions for a completed run.

    :param run: Run identifier or existing run directory.
    :type run: str
    :param split: Test split or validation split for reproduction checks.
    :type split: str
    :param as_json: Emit exactly one JSON evaluation summary.
    :type as_json: bool
    :raises typer.Exit: If arguments, run artifacts, or evaluation are invalid.
    """
    try:
        if split not in {'test', 'val'}:
            raise ConfigError('Evaluation split must be test or val.')
        paths = get_paths()
        result = evaluate(run_dir=resolve_run(paths=paths, run=run), paths=paths, split=split)
    except (ConfigError, OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else 130
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='evaluate').error('evaluation_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    if as_json:
        emit(result=result, as_json=True)
    else:
        typer.echo(message='observation  accuracy  95% Wilson interval')
        for row in result['rows']:
            typer.echo(
                message=f'{row["t"]:>11}  {row["accuracy"]:.2%}  '
                f'[{row["ci_low"]:.2%}, {row["ci_high"]:.2%}]'
            )
        typer.echo(message=f'Evaluation: {result["eval_dir"]}')

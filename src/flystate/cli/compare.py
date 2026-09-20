"""Compare paired evaluations with strict compatibility diagnostics."""

import os
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INCOMPATIBLE, RUNTIME_ERROR, emit
from flystate.evaluation.compare import IncompatibleRunsError, compare, comparison_markdown
from flystate.experiments.config import ConfigError
from flystate.log import get_logger
from flystate.settings import get_paths, output_path
from flystate.storage.runs import resolve_run


def compare_command(
    run_a: str,
    run_b: str,
    eval_a: Annotated[str | None, typer.Option('--eval-a')] = None,
    eval_b: Annotated[str | None, typer.Option('--eval-b')] = None,
    allow: Annotated[str, typer.Option('--allow')] = 'memory.mode,name',
    bootstrap: Annotated[int | None, typer.Option('--bootstrap')] = None,
    seed: Annotated[int | None, typer.Option('--seed')] = None,
    markdown: Annotated[Path | None, typer.Option('--markdown')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Compare accuracy with paired bootstrap intervals and McNemar tests.

    :param run_a: First run ID or existing path.
    :type run_a: str
    :param run_b: Second run ID or existing path.
    :type run_b: str
    :param eval_a: Optional explicit evaluation ID for A.
    :type eval_a: Optional[str]
    :param eval_b: Optional explicit evaluation ID for B.
    :type eval_b: Optional[str]
    :param allow: Comma-separated exact dotted configuration keys permitted to differ.
    :type allow: str
    :param bootstrap: Optional positive resample count.
    :type bootstrap: Optional[int]
    :param seed: Optional nonnegative resampling seed.
    :type seed: Optional[int]
    :param markdown: Optional new Markdown report path within the data home.
    :type markdown: Optional[Path]
    :param as_json: Emit exactly one JSON result or diagnostic.
    :type as_json: bool
    :raises typer.Exit: If configuration, compatibility, or report storage fails.
    """
    try:
        if (bootstrap is not None and bootstrap < 1) or (seed is not None and seed < 0):
            raise ConfigError('Bootstrap count must be positive and seed nonnegative.')
        paths = get_paths()
        destination = None if markdown is None else output_path(path=markdown, paths=paths)
        if destination is not None and destination.exists():
            raise FileExistsError(f'Markdown report already exists: {destination}')
        result = compare(
            run_a=resolve_run(paths=paths, run=run_a),
            run_b=resolve_run(paths=paths, run=run_b),
            paths=paths,
            eval_a=eval_a,
            eval_b=eval_b,
            allowed=tuple(key.strip() for key in allow.split(',') if key.strip()),
            bootstrap=bootstrap,
            seed=seed,
        )
        report = comparison_markdown(result=result)
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=destination.parent, delete_on_close=False
            ) as stream:
                stream.write(report)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                Path(stream.name).replace(target=destination)
    except (ConfigError, OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        code = (
            INCOMPATIBLE
            if isinstance(error, IncompatibleRunsError)
            else CONFIG_ERROR
            if isinstance(error, ConfigError)
            else 130
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='compare').error('comparison_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    if as_json:
        emit(result=result, as_json=True)
    else:
        typer.echo(message=report.rstrip())

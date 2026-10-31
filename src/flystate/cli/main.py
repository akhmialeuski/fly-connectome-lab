"""The public flystate command-line application."""

from dataclasses import asdict
from typing import Annotated

import typer

from flystate import __version__
from flystate.cli.common import emit
from flystate.log import configure_logging
from flystate.settings import get_paths

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    verbose: Annotated[int, typer.Option('--verbose', '-v', count=True)] = 0,
    quiet: Annotated[bool, typer.Option('--quiet', '-q')] = False,
    version: Annotated[bool, typer.Option('--version', is_eager=True)] = False,
) -> None:
    """Run reproducible MaleCNS sequential visual memory experiments.

    :param verbose: Verbosity increment.
    :type verbose: int
    :param quiet: Suppress informational logs.
    :type quiet: bool
    :param version: Display the installed package version.
    :type version: bool
    """
    if version:
        typer.echo(message=__version__)
        raise typer.Exit()
    configure_logging(verbosity=-1 if quiet else verbose)


def group_help() -> None:
    """Inspect the available commands in this group."""


for group_name in ('brain', 'dataset', 'episode', 'experiment', 'trace'):
    group = typer.Typer(no_args_is_help=True, help=f'{group_name.capitalize()} commands.')
    group.callback()(group_help)
    app.add_typer(typer_instance=group, name=group_name)


@app.command(name='paths')
def paths_command(as_json: Annotated[bool, typer.Option('--json')] = False) -> None:
    """Show the resolved experiment storage paths.

    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    emit(
        result={key: str(value) for key, value in asdict(obj=get_paths()).items()}, as_json=as_json
    )

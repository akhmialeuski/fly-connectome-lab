"""The public flystate command-line application."""

from dataclasses import asdict
from typing import Annotated

import typer

from flystate import __version__
from flystate.cli.brain import app as brain_app
from flystate.cli.common import emit
from flystate.cli.compare import compare_command
from flystate.cli.dataset import app as dataset_app
from flystate.cli.diagnose import app as diagnose_app
from flystate.cli.doctor import doctor_command
from flystate.cli.episode import app as episode_app
from flystate.cli.evaluate import evaluate_command
from flystate.cli.experiment import app as experiment_app
from flystate.cli.serve import serve_command
from flystate.cli.trace import app as trace_app
from flystate.cli.train import train_command
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


app.add_typer(typer_instance=trace_app, name='trace')
app.add_typer(typer_instance=experiment_app, name='experiment')
app.add_typer(typer_instance=brain_app, name='brain')
app.add_typer(typer_instance=dataset_app, name='dataset')
app.add_typer(typer_instance=diagnose_app, name='diagnose')
app.add_typer(typer_instance=episode_app, name='episode')
app.command(name='doctor')(doctor_command)
app.command(name='train')(train_command)
app.command(name='evaluate')(evaluate_command)
app.command(name='compare')(compare_command)
app.command(name='serve')(serve_command)


@app.command(name='paths')
def paths_command(as_json: Annotated[bool, typer.Option('--json')] = False) -> None:
    """Show the resolved experiment storage paths.

    :param as_json: Emit one JSON object.
    :type as_json: bool
    """
    emit(
        result={key: str(value) for key, value in asdict(obj=get_paths()).items()}, as_json=as_json
    )

"""Experiment configuration command adapters."""

from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, emit
from flystate.experiments.config import ConfigError, config_hash, effective_yaml, load_config
from flystate.log import get_logger

app = typer.Typer(no_args_is_help=True, help='Validate experiment configurations.')


@app.command(name='validate')
def validate_command(
    config: Annotated[Path, typer.Argument(help='Experiment YAML path.')],
    overrides: Annotated[
        list[str] | None, typer.Option('--set', help='Override dotted.path=value.')
    ] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Validate an experiment and show its complete configuration and hash.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Ordered field overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit exactly one JSON object on stdout.
    :type as_json: bool
    :raises typer.Exit: If configuration validation fails.
    """
    try:
        cfg = load_config(path=config, overrides=overrides or ())
    except ConfigError as error:
        get_logger(name='experiment').error('invalid_config', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=CONFIG_ERROR) from error
    emit(
        result={'config_hash': config_hash(cfg=cfg), 'effective_yaml': effective_yaml(cfg=cfg)},
        as_json=as_json,
    )

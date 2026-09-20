"""Train portable per-observation readouts from an existing completed trace cache."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, RUNTIME_ERROR, emit
from flystate.datasets.errors import DatasetError
from flystate.experiments.config import ConfigError, load_config
from flystate.log import LOGGER_NAMESPACE, get_logger
from flystate.readouts.training import train
from flystate.settings import get_paths


def train_command(
    config: Path,
    overrides: Annotated[list[str] | None, typer.Option('--set')] = None,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Fit a linear readout per observation using train data and report validation accuracy.

    :param config: Experiment YAML path.
    :type config: Path
    :param overrides: Optional dotted configuration overrides.
    :type overrides: Optional[list[str]]
    :param as_json: Emit one JSON run summary.
    :type as_json: bool
    :raises typer.Exit: If configuration, fitting, storage, or interruption fails.
    """
    try:
        cfg = load_config(path=config, overrides=overrides or ())
        level = logging.getLogger(name=LOGGER_NAMESPACE).getEffectiveLevel()
        result = train(
            cfg=cfg,
            paths=get_paths(),
            original_yaml=config.read_text(encoding='utf-8'),
            verbosity=-1 if level >= logging.WARNING else int(level == logging.DEBUG),
        )
    except (
        ConfigError,
        DatasetError,
        OSError,
        ValueError,
        RuntimeError,
        Warning,
        KeyboardInterrupt,
    ) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else 130
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='train').error('training_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    if as_json:
        emit(result=result, as_json=True)
    else:
        typer.echo(message='observation  validation accuracy')
        for step, accuracy in result['val_accuracy_by_t'].items():
            typer.echo(message=f'{step:>11}  {accuracy:.2%}')
        typer.echo(message=f'Run: {result["run_id"]}')

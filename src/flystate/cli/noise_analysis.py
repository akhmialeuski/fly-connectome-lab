"""CLI entry point for the fixed six-readout episode-noise comparison."""

import json
from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.noise_analysis import analyze_noise_recognition
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def noise_analysis_command(
    config: Path,
    schedule: Annotated[Path, typer.Option('--schedule', help='Frozen trace and fit paths JSON.')],
    output: Annotated[Path, typer.Option('--output', help='Fresh analysis directory.')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Analyze fixed development-only noise and precision readouts.

    :param config: Original persistent effective experiment configuration.
    :type config: Path
    :param schedule: Frozen study path mapping and membership hash.
    :type schedule: Path
    :param output: New immutable analysis directory inside FLYSTATE_HOME.
    :type output: Path
    :param as_json: Emit exactly one JSON result on stdout.
    :type as_json: bool
    :raises typer.Exit: If schedule, source evidence, or analysis fails.
    """
    try:
        cfg = load_config(path=config)
        try:
            frozen = json.loads(s=schedule.read_text(encoding='utf-8'))
        except json.JSONDecodeError as error:
            raise ConfigError('Schedule must be valid JSON.') from error
        if (
            not isinstance(frozen, dict)
            or not isinstance(frozen.get('membership_sha256'), str)
            or not isinstance(frozen.get('historical'), str)
            or not isinstance(frozen.get('traces'), dict)
            or not isinstance(frozen.get('probes'), dict)
        ):
            raise ConfigError('Schedule requires frozen membership, historical, traces and probes.')
        result = analyze_noise_recognition(
            cfg=cfg, paths=get_paths(), output=output, schedule=frozen
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='noise_analysis').error('noise_analysis_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

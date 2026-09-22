"""Explicit entry point for frozen training-only signal-stability measurements."""

import json
from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.stability import run_stability
from flystate.experiments.config import ConfigError, load_config
from flystate.log import get_logger
from flystate.settings import get_paths


def stability_command(
    config: Path,
    membership: Annotated[
        Path, typer.Option('--membership', help='Frozen sample IDs and SHA-256 JSON.')
    ],
    output: Annotated[Path, typer.Option('--output', help='Fresh directory inside FLYSTATE_HOME.')],
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Measure native precision and paired noise responses without fitting a classifier.

    :param config: Original persistent, noise-enabled effective configuration.
    :type config: Path
    :param membership: JSON object with sample_ids and sample_ids_sha256.
    :type membership: Path
    :param output: New immutable diagnostic output directory.
    :type output: Path
    :param as_json: Emit exactly one JSON result on stdout.
    :type as_json: bool
    :raises typer.Exit: If arguments or measurements fail.
    """
    try:
        cfg = load_config(path=config)
        frozen = json.loads(s=membership.read_text(encoding='utf-8'))
        if (
            not isinstance(frozen, dict)
            or not isinstance(frozen.get('sample_ids'), list)
            or not all(isinstance(s, str) for s in frozen['sample_ids'])
            or not isinstance(frozen.get('sample_ids_sha256'), str)
        ):
            raise ConfigError('Membership requires a list of sample IDs and its SHA-256 string.')
        result = run_stability(
            cfg=cfg,
            paths=get_paths(),
            output=output,
            sample_ids=frozen['sample_ids'],
            membership_sha256=frozen['sample_ids_sha256'],
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else (INTERRUPTED if isinstance(error, KeyboardInterrupt) else RUNTIME_ERROR)
        )
        get_logger(name='stability').error('stability_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

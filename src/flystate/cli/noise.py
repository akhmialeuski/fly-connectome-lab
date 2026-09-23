"""CLI entry point for immutable native episode-noise traces."""

import json
from pathlib import Path
from typing import Annotated

import typer

from flystate.cli.common import CONFIG_ERROR, INTERRUPTED, RUNTIME_ERROR, emit
from flystate.diagnostics.noise import generate_noise_trace
from flystate.experiments.config import ConfigError, load_config
from flystate.hashing import sha256_obj
from flystate.log import get_logger
from flystate.settings import get_paths


def noise_trace_command(
    config: Path,
    membership: Annotated[Path, typer.Option('--membership', help='Frozen split membership JSON.')],
    output: Annotated[Path, typer.Option('--output', help='Fresh directory inside FLYSTATE_HOME.')],
    episode_seed: Annotated[int, typer.Option('--episode-seed', help='Episode-noise seed.')],
    noise_enabled: Annotated[
        bool, typer.Option('--noise-enabled/--noise-disabled', help='Apply episode noise.')
    ] = True,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Generate native sequential responses for one fixed noise condition.

    :param config: Original persistent, noise-enabled effective configuration.
    :type config: Path
    :param membership: Frozen train/validation sample IDs and canonical hash.
    :type membership: Path
    :param output: New immutable trace attempt directory.
    :type output: Path
    :param episode_seed: Nonnegative episode-noise seed, independent of warmup.
    :type episode_seed: int
    :param noise_enabled: Whether episode noise remains enabled after warmup.
    :type noise_enabled: bool
    :param as_json: Emit exactly one JSON result on stdout.
    :type as_json: bool
    :raises typer.Exit: If arguments are invalid or the trace attempt fails.
    """
    try:
        cfg = load_config(path=config)
        try:
            frozen = json.loads(s=membership.read_text(encoding='utf-8'))
        except json.JSONDecodeError as error:
            raise ConfigError('Membership must be valid JSON.') from error
        groups = frozen.get('sample_ids_by_split') if isinstance(frozen, dict) else None
        if (
            episode_seed < 0
            or not isinstance(groups, dict)
            or set(groups) != {'train', 'val'}
            or not all(isinstance(rows, list) for rows in groups.values())
            or not all(isinstance(sample_id, str) for rows in groups.values() for sample_id in rows)
            or any(not rows or rows != sorted(set(rows)) for rows in groups.values())
            or sha256_obj(obj=groups) != frozen.get('membership_sha256')
        ):
            raise ConfigError('Require sorted unique train/val membership and matching SHA-256.')
        result = generate_noise_trace(
            cfg=cfg,
            paths=get_paths(),
            output=output,
            membership=groups,
            membership_sha256=frozen['membership_sha256'],
            episode_seed=episode_seed,
            noise_enabled=noise_enabled,
        )
    except (Exception, KeyboardInterrupt) as error:
        code = (
            CONFIG_ERROR
            if isinstance(error, ConfigError)
            else INTERRUPTED
            if isinstance(error, KeyboardInterrupt)
            else RUNTIME_ERROR
        )
        get_logger(name='noise_trace').error('noise_trace_failed', detail=str(error))
        if as_json:
            emit(result={'error': str(error)}, as_json=True)
        raise typer.Exit(code=code) from error
    emit(result=result, as_json=as_json)

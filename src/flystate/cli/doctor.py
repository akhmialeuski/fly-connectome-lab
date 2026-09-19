"""Root command for local installation diagnostics."""

from dataclasses import asdict
from typing import Annotated

import typer

from flystate.cli.common import RUNTIME_ERROR, emit
from flystate.doctor import run_checks
from flystate.settings import get_paths


def doctor_command(
    no_hash: Annotated[
        bool, typer.Option('--no-hash', help='Skip brain checksum verification.')
    ] = False,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Check local dependencies, storage, CPU runtime, and registered data.

    :param no_hash: Skip brain checksum computation.
    :type no_hash: bool
    :param as_json: Emit the ordered checks in one JSON object.
    :type as_json: bool
    :raises typer.Exit: If any diagnostic has error status.
    """
    checks = run_checks(paths=get_paths(), check_hash=not no_hash)
    if as_json:
        emit(result={'checks': [asdict(obj=check) for check in checks]}, as_json=True)
    else:
        emit(
            result={check.name: f'{check.status}: {check.detail}' for check in checks},
            as_json=False,
        )
    if any(check.status == 'error' for check in checks):
        raise typer.Exit(code=RUNTIME_ERROR)

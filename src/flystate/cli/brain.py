"""Brain acquisition and inspection commands."""

from typing import Annotated, Never
from zipfile import BadZipFile

import flybrain
import typer

from flystate.brain import files
from flystate.cli.common import RUNTIME_ERROR, emit
from flystate.log import get_logger
from flystate.settings import get_paths

app = typer.Typer(no_args_is_help=True, help='Download and inspect connectome files.')


def _report(check_hash: bool, require_published: bool, as_json: bool) -> bool:
    """Print one integrity and metadata report with a meaningful exit status.

    :param check_hash: Compute file digests.
    :type check_hash: bool
    :param require_published: Require production neuron and connection counts.
    :type require_published: bool
    :param as_json: Emit machine-readable output.
    :type as_json: bool
    :returns: Whether all requested integrity checks pass.
    :rtype: bool
    """
    paths = get_paths()
    verification = files.verify_brain_files(brain_dir=paths.brain, check_hash=check_hash)
    valid = all(item['exists'] and item['sha256_ok'] is not False for item in verification.values())
    present = all(item['exists'] for item in verification.values())
    stats = files.brain_stats(brain_dir=paths.brain) if present else {}
    if require_published:
        valid = (
            valid
            and stats.get('neurons') == files.EXPECTED_NEURONS
            and (stats.get('connections') == files.EXPECTED_CONNECTIONS)
        )
    emit(
        result={
            'directory': str(paths.brain),
            'files': verification,
            'flybrain_version': flybrain.__version__,
            'stats': stats,
            'valid': valid,
        },
        as_json=as_json,
    )
    return valid


def _failure(error: Exception, as_json: bool) -> Never:
    """Report an expected file operation failure consistently.

    :param error: Operational exception.
    :type error: Exception
    :param as_json: Emit JSON in addition to the stderr log.
    :type as_json: bool
    :returns: Never returns normally.
    :rtype: Never
    :raises typer.Exit: Always exits with the runtime error code.
    """
    get_logger(name='brain').error('brain_files_failed', detail=str(error))
    if as_json:
        emit(result={'error': str(error)}, as_json=True)
    raise typer.Exit(code=RUNTIME_ERROR) from error


@app.command(name='download')
def download_command(
    force: Annotated[bool, typer.Option('--force', help='Replace existing brain files.')] = False,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Download and verify the published MaleCNS brain into FLYSTATE_HOME.

    :param force: Replace existing files.
    :type force: bool
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    try:
        files.download_brain(paths=get_paths(), force=force)
        valid = _report(check_hash=True, require_published=True, as_json=as_json)
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, EOFError) as error:
        _failure(error=error, as_json=as_json)
    if not valid:
        raise typer.Exit(code=RUNTIME_ERROR)


@app.command(name='info')
def info_command(
    no_hash: Annotated[bool, typer.Option('--no-hash', help='Skip SHA-256 computation.')] = False,
    as_json: Annotated[bool, typer.Option('--json')] = False,
) -> None:
    """Inspect installed brain files without downloading or creating a simulator.

    :param no_hash: Skip digest verification.
    :type no_hash: bool
    :param as_json: Emit one JSON report.
    :type as_json: bool
    """
    try:
        valid = _report(check_hash=not no_hash, require_published=False, as_json=as_json)
    except (OSError, ValueError, RuntimeError, KeyError, BadZipFile, EOFError) as error:
        _failure(error=error, as_json=as_json)
    if not valid:
        raise typer.Exit(code=RUNTIME_ERROR)

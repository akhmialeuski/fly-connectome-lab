"""Start the local, read-only experiment results viewer."""

from typing import Annotated

import typer
import uvicorn

from flystate.cli.common import emit
from flystate.settings import get_paths
from flystate.viewer.app import create_app


def serve_command(
    port: Annotated[int, typer.Option(min=1024, max=65535, help='Local HTTP port.')] = 8765,
    as_json: Annotated[bool, typer.Option('--json', help='Emit one server descriptor.')] = False,
) -> None:
    """Browse saved experiments at http://127.0.0.1:8765; stop with Ctrl+C.

    :param port: Loopback listening port.
    :type port: int
    :param as_json: Emit one JSON startup descriptor on stdout.
    :type as_json: bool
    """
    paths = get_paths()
    emit(
        result={'url': f'http://127.0.0.1:{port}', 'home': str(paths.home), 'read_only': True},
        as_json=as_json,
    )
    uvicorn.run(
        app=create_app(paths=paths), host='127.0.0.1', port=port, log_config=None, access_log=False
    )

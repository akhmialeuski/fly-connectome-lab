"""Loopback-only HTTP interface to immutable experiment artifacts."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from flystate.experiments.config import ConfigError
from flystate.settings import Paths
from flystate.viewer.repository import Repository


def repository(request: Request) -> Repository:
    """Resolve the repository owned by this application instance.

    :param request: Incoming HTTP request.
    :type request: Request
    :returns: Read-only artifact repository.
    :rtype: Repository
    """
    return request.app.state.repository


Store = Annotated[Repository, Depends(repository)]


def catalog(store: Store) -> dict[str, Any]:
    """Discover local artifacts.

    :param store: Artifact repository.
    :type store: Repository
    :returns: Catalog with recoverable warnings.
    :rtype: dict[str, Any]
    """
    return store.catalog()


def run(run_id: str, store: Store) -> dict[str, Any]:
    """Read a run and its evaluations.

    :param run_id: Recorded run identifier.
    :type run_id: str
    :param store: Artifact repository.
    :type store: Repository
    :returns: Run details.
    :rtype: dict[str, Any]
    """
    return store.run(run_id=run_id)


def evaluation(run_id: str, eval_id: str, store: Store) -> dict[str, Any]:
    """Read completed evaluation tables.

    :param run_id: Recorded run identifier.
    :type run_id: str
    :param eval_id: Recorded evaluation identifier.
    :type eval_id: str
    :param store: Artifact repository.
    :type store: Repository
    :returns: Evaluation details.
    :rtype: dict[str, Any]
    """
    return store.evaluation(run_id=run_id, eval_id=eval_id)


def predictions(
    run_id: str,
    eval_id: str,
    store: Store,
    t: Annotated[int | None, Query(ge=1)] = None,
    correct: bool | None = None,
    sample_id: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Read a filtered prediction page.

    :param run_id: Recorded run identifier.
    :type run_id: str
    :param eval_id: Recorded evaluation identifier.
    :type eval_id: str
    :param store: Artifact repository.
    :type store: Repository
    :param t: One-based observation filter.
    :type t: Optional[int]
    :param correct: Correctness filter.
    :type correct: Optional[bool]
    :param sample_id: Exact sample identifier filter.
    :type sample_id: Optional[str]
    :param offset: Nonnegative row offset.
    :type offset: int
    :param limit: Maximum records per response.
    :type limit: int
    :returns: Prediction page and matching count.
    :rtype: dict[str, Any]
    """
    return store.predictions(
        run_id=run_id,
        eval_id=eval_id,
        t=t,
        correct=correct,
        sample_id=sample_id,
        offset=offset,
        limit=limit,
    )


def report(kind: str, name: str, store: Store) -> dict[str, Any]:
    """Read a recorded report.

    :param kind: Report category.
    :type kind: str
    :param name: Report basename.
    :type name: str
    :param store: Artifact repository.
    :type store: Repository
    :returns: Structured report or plain text.
    :rtype: dict[str, Any]
    """
    return store.report(kind=kind, name=name)


def sample(run_id: str, sample_id: str, store: Store) -> dict[str, Any]:
    """Read cached neural activity and stimulus boxes.

    :param run_id: Recorded run identifier.
    :type run_id: str
    :param sample_id: Recorded sample identifier.
    :type sample_id: str
    :param store: Artifact repository.
    :type store: Repository
    :returns: Cached episode.
    :rtype: dict[str, Any]
    """
    return store.sample(run_id=run_id, sample_id=sample_id)


def sample_image(run_id: str, sample_id: str, store: Store) -> Response:
    """Serve a prepared local image as PNG.

    :param run_id: Recorded run identifier.
    :type run_id: str
    :param sample_id: Recorded sample identifier.
    :type sample_id: str
    :param store: Artifact repository.
    :type store: Repository
    :returns: PNG response.
    :rtype: Response
    """
    return Response(content=store.image(run_id=run_id, sample_id=sample_id), media_type='image/png')


def experiment(path: str, store: Store) -> dict[str, Any]:
    """Read a manifest-backed experiment at any depth.

    :param path: Relative experiment directory in the catalog.
    :type path: str
    :param store: Read-only artifact repository.
    :type store: Repository
    :returns: Available documents, evidence, and identity references.
    :rtype: dict[str, Any]
    """
    return store.experiment_detail(relative=path)


def evidence(
    path: str,
    name: str,
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Read a bounded saved evidence preview.

    :param path: Relative experiment directory.
    :type path: str
    :param name: Relative evidence filename.
    :type name: str
    :param store: Read-only artifact repository.
    :type store: Repository
    :param offset: Nonnegative table offset.
    :type offset: int
    :param limit: Maximum table rows, bounded by 200.
    :type limit: int
    :returns: Text or a table page.
    :rtype: dict[str, Any]
    """
    return store.experiments.evidence(relative=path, name=name, offset=offset, limit=limit)


def experiment_image(path: str, sample_id: str, store: Store) -> Response:
    """Serve an existing image recorded in generic experiment membership.

    :param path: Relative experiment directory.
    :type path: str
    :param sample_id: Recorded sample identifier.
    :type sample_id: str
    :param store: Read-only artifact repository.
    :type store: Repository
    :returns: Prepared PNG response.
    :rtype: Response
    """
    return Response(
        content=store.experiment_image(relative=path, sample_id=sample_id), media_type='image/png'
    )


async def artifact_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate missing or damaged artifacts into usable browser errors.

    :param request: Incoming request.
    :type request: Request
    :param exc: Artifact read failure.
    :type exc: Exception
    :returns: Error object with missing or conflict status.
    :rtype: JSONResponse
    """
    return JSONResponse(
        content={'detail': str(exc)},
        status_code=404 if isinstance(exc, FileNotFoundError) else 409,
    )


async def response_headers(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Prevent stale artifact reads and cross-origin embedding of local results.

    :param request: Incoming HTTP request.
    :type request: Request
    :param call_next: Next middleware or application handler.
    :type call_next: RequestResponseEndpoint
    :returns: Response with local viewer policies.
    :rtype: Response
    """
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'none'"
    )
    return response


def create_app(paths: Paths) -> FastAPI:
    """Create an isolated viewer with bundled assets and no write routes.

    :param paths: Existing experiment storage installation.
    :type paths: Paths
    :returns: Local results application.
    :rtype: FastAPI
    """
    app = FastAPI(title='Flystate Results', docs_url=None, redoc_url=None, openapi_url=None)
    app.state.repository = Repository(paths=paths)
    app.add_middleware(BaseHTTPMiddleware, dispatch=response_headers)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]'])
    for error in (OSError, ValueError, KeyError, TypeError, ConfigError):
        app.add_exception_handler(exc_class_or_status_code=error, handler=artifact_error)
    app.get('/api/catalog')(catalog)
    app.get('/api/experiments/detail')(experiment)
    app.get('/api/experiments/evidence')(evidence)
    app.get('/api/experiments/image')(experiment_image)
    app.get('/api/runs/{run_id}')(run)
    prefix = '/api/runs/{run_id}/evaluations/{eval_id}'
    app.get(prefix)(evaluation)
    app.get(prefix + '/predictions')(predictions)
    app.get('/api/reports/{kind}/{name}')(report)
    app.get('/api/runs/{run_id}/samples/{sample_id}')(sample)
    app.get('/api/runs/{run_id}/samples/{sample_id}/image')(sample_image)
    app.mount(
        '/', StaticFiles(directory=Path(__file__).parent / 'static', html=True), name='assets'
    )
    return app

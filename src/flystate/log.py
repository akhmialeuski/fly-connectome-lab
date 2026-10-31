"""One logging entry point for console and JSON Lines destinations."""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

LOGGER_NAMESPACE: str = 'flystate'


def configure_logging(verbosity: int = 0, json_path: Path | None = None) -> None:
    """Replace project handlers and configure structured logging.

    :param verbosity: Negative for warnings, zero for information, positive for debug.
    :type verbosity: int
    :param json_path: Optional append-only JSON Lines destination.
    :type json_path: Optional[Path]
    """
    level = logging.WARNING if verbosity < 0 else logging.DEBUG if verbosity > 0 else logging.INFO
    logger = logging.getLogger(name=LOGGER_NAMESPACE)
    for handler in logger.handlers[:]:
        logger.removeHandler(hdlr=handler)
        handler.close()
    logger.setLevel(level=level)
    logger.propagate = False
    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso', utc=True),
    ]
    destinations: list[tuple[logging.Handler, Any]] = [
        (logging.StreamHandler(stream=sys.stderr), structlog.dev.ConsoleRenderer(colors=False))
    ]
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        destinations.append(
            (
                logging.FileHandler(filename=json_path, encoding='utf-8'),
                structlog.processors.JSONRenderer(),
            )
        )
    for handler, renderer in destinations:
        destination_processors: list[Any] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta
        ]
        if isinstance(renderer, structlog.dev.ConsoleRenderer):
            destination_processors.append(
                structlog.processors.TimeStamper(fmt='%H:%M:%S', utc=True)
            )
        destination_processors.append(renderer)
        handler.setFormatter(
            fmt=structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=processors,
                processors=destination_processors,
            )
        )
        logger.addHandler(hdlr=handler)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger within the project namespace.

    :param name: Component name.
    :type name: str
    :returns: Logger supporting structured bound context.
    :rtype: structlog.stdlib.BoundLogger
    """
    return structlog.get_logger(f'{LOGGER_NAMESPACE}.{name}')

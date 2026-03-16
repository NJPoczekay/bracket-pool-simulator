"""Structured logging configuration for simulation runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from structlog.stdlib import BoundLogger
from structlog.types import Processor


def configure_structured_logging(*, level: str, log_path: Path | None = None) -> BoundLogger:
    """Configure JSON logging to stderr and an optional log file."""

    processors: list[Processor] = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(sort_keys=True),
    ]
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    logger = logging.getLogger("bracket_sim")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(_to_logging_level(level))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stderr_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

    return BoundLogger(logger, processors, {})


def _to_logging_level(level: str) -> int:
    mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    return mapping[level]

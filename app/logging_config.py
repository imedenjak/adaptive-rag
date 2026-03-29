"""
Structured logging configuration using structlog.

Local dev  (LOG_FORMAT=dev):  colored, human-readable console output
Production (LOG_FORMAT=json): JSON lines to stdout for log aggregators
                               (Datadog, CloudWatch, Loki, etc.)

Usage in any module:
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("event.name", key="value")

Bind request-scoped context once (e.g., per query) and it propagates to all
subsequent log calls in that thread:
    structlog.contextvars.bind_contextvars(query_id="abc123")
    structlog.contextvars.clear_contextvars()
"""
import logging
import sys

import structlog

from .config import LOG_FORMAT, LOG_LEVEL


def configure_logging() -> None:
    # 1. Configure stdlib root logger so third-party libraries (langchain,
    #    httpx, qdrant-client) are captured at the right level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    )
    # Quiet noisy third-party loggers that aren't useful day-to-day.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    shared_processors: list = [
        # Merge any context bound via bind_contextvars() (e.g., query_id).
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if LOG_FORMAT == "json":
        # Production: one JSON object per line, easy for log aggregators.
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Dev: pretty colored output with aligned columns.
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

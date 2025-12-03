"""Centralized logging setup for OmniCLI."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


LOGGER_NAME = "omnicli"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a Rich handler (idempotent)."""
    if logging.getLogger(LOGGER_NAME).handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a namespaced logger under the OmniCLI hierarchy."""
    base = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(base)




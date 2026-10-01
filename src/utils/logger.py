"""Rich-based logging configuration shared by every AeroTrace-RCA component."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console(stderr=True)
_CONFIGURED = False
_ROOT_LOGGER_NAME = "aerotrace"


def get_console() -> Console:
    """Return the shared Rich console used for log output."""
    return _CONSOLE


def setup_logging(level: str | int = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the ``aerotrace`` logger hierarchy.

    Args:
        level: Logging level name or numeric value.
        log_file: Optional file that receives a plain-text copy of every record.

    Returns:
        The configured root ``aerotrace`` logger.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    if not _CONFIGURED:
        handler = RichHandler(
            console=_CONSOLE,
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            log_time_format="[%H:%M:%S]",
        )
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        # Quieten chatty third-party libraries.
        for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "matplotlib", "anthropic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True

    for handler in logger.handlers:
        handler.setLevel(numeric_level)

    if log_file is not None and not any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
        for h in logger.handlers
    ):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``aerotrace`` hierarchy.

    Args:
        name: Component name, e.g. ``"agents.supervisor"``.
    """
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")

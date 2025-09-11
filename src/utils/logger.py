"""Centralized logging configuration using Loguru.

Provides a consistent, structured logging interface across all modules.
Supports file rotation, colored terminal output, and structured fields.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    colorize: bool = True,
) -> None:
    """Configure the global Loguru logger.

    Sets up console and optional file handlers with structured formatting.
    Removes the default Loguru handler and replaces with project-specific config.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write structured logs to disk.
        rotation: Log file rotation policy (e.g., "10 MB", "1 day").
        retention: How long to retain rotated logs.
        colorize: Whether to colorize terminal output.

    Example:
        >>> setup_logger(level="DEBUG", log_file=Path("logs/run.log"))
    """
    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            compression="gz",
            serialize=False,
            enqueue=True,  # Thread-safe async writing
        )
        logger.info(f"File logging enabled: {log_file}")


def get_logger(name: str) -> "logger":
    """Get a module-scoped logger.

    Args:
        name: Module name (typically __name__).

    Returns:
        Loguru logger bound with module context.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Starting module")
    """
    return logger.bind(module=name)


def get_trace_logger(name: str) -> "logger":
    """Get a TRACE-level logger for verbose debugging (disabled in production).

    Args:
        name: Module name (typically __name__).

    Returns:
        Loguru logger bound with module context at TRACE level.
    """
    return logger.bind(module=name, level="TRACE")

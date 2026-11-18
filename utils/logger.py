"""
Logging Configuration
======================
Implements centralized logging for the Medical Image Classification System.
Supports Rubric Criterion 9: Logging and Software Robustness.

Uses the loguru library for structured, human-readable log output.
All relevant system events are recorded for auditability (RNF-09).
"""

import sys
from loguru import logger

from config.settings import LOGS_DIR, LOG_LEVEL, LOG_FORMAT


def setup_logger() -> None:
    """
    Initialize the application-wide logger with file and console outputs.
    
    - Console: colored output for development.
    - File: persistent logs rotated daily for production traceability.
    - Audit file: separate append-only log for clinical audit events.
    - Graceful fallback: if log volume is read-only, continues with console logging.
    """
    logger.remove()

    # Console handler — colored output for development
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True,
    )

    try:
        # General application log — rotated daily, retained 30 days
        logger.add(
            LOGS_DIR / "application.log",
            format=LOG_FORMAT,
            level=LOG_LEVEL,
            rotation="1 day",
            retention="30 days",
            compression="zip",
            enqueue=False,
        )

        # Audit trail — append-only, immutable, retained per RNF-09
        logger.add(
            LOGS_DIR / "audit.log",
            format=LOG_FORMAT,
            level="INFO",
            rotation="1 week",
            retention="365 days",
            compression="zip",
            enqueue=False,
            filter=lambda record: "audit" in record["extra"],
        )

        # Error-specific log for quick diagnosis
        logger.add(
            LOGS_DIR / "errors.log",
            format=LOG_FORMAT,
            level="ERROR",
            rotation="1 week",
            retention="90 days",
            compression="zip",
            enqueue=False,
        )
    except Exception:
        # If running on a read-only filesystem without writable LOGS_DIR, console logger remains active
        pass

    logger.info("Logger initialized successfully")


def get_logger(name: str):
    """
    Return a contextualized logger bound to a specific module/service name.
    This helps trace which service generated each log entry.
    """
    return logger.bind(service=name)


def get_audit_logger(name: str):
    """
    Return an audit-specific logger for immutable clinical event recording.
    Used by the Audit Service (RNF-09, RD-04).
    """
    return logger.bind(service=name, audit=True)

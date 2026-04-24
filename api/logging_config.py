"""
Structured logging configuration using loguru.

Replaces all print() debugging with proper structured logs.
Logs to both console (human-readable) and file (JSON for log aggregation).
"""
import sys
import logging
from pathlib import Path

from loguru import logger


# ── Intercept stdlib logging (uvicorn, sqlalchemy, etc.) ──────────────────

class InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru for unified output."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_dir: str = "logs", debug: bool = False) -> None:
    """
    Configure loguru for the enterprise RAG platform.
    
    - Console: colored, human-readable (INFO+ in prod, DEBUG in dev)
    - File: JSON-structured, rotated daily, kept 30 days
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default loguru handler
    logger.remove()

    # ── Console sink (human-readable) ─────────────────────────────────
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=debug,
    )

    # ── JSON file sink (for log aggregation on DGX) ───────────────────
    logger.add(
        str(log_path / "rag_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{message}",
        serialize=True,          # JSON output
        rotation="00:00",        # Rotate at midnight
        retention="30 days",
        compression="gz",
        enqueue=True,            # Thread-safe async writes
    )

    # ── Error-only file (quick triage) ────────────────────────────────
    logger.add(
        str(log_path / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} — {message}",
        rotation="00:00",
        retention="90 days",
        compression="gz",
        enqueue=True,
    )

    # ── Intercept stdlib loggers ──────────────────────────────────────
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error",
                        "sqlalchemy.engine", "fastapi"):
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers = [InterceptHandler()]
        stdlib_logger.propagate = False

    logger.info("Logging initialized — console + file output active")

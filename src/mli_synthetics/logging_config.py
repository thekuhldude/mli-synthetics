"""Loguru configuration."""
import sys
from pathlib import Path

from loguru import logger

_configured = False


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "mli_synthetics_{time:YYYYMMDD}.log",
            level="DEBUG",
            rotation="10 MB",
            retention="14 days",
            encoding="utf-8",
        )
    _configured = True


def get_logger():
    if not _configured:
        configure_logging()
    return logger

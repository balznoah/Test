"""Rotating file + console logger."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(
        logs_dir / "pipeline.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    eh = RotatingFileHandler(
        logs_dir / "errors.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    return logger

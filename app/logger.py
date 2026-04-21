"""
Logging configuration for the Pneumonia Detection AI application.

Provides centralized logging setup with consistent formatting across the app.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config import LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure and return a logger with the specified name.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to write logs to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger for app-wide use
app_logger = setup_logger(__name__)

"""
Logging utilities for the Dad Cam Pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "dad_cam_pipeline",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up the global logger.

    Args:
        name: Logger name
        log_dir: Directory for log files (optional)
        level: Logging level
        console: Whether to output to console

    Returns:
        Configured logger instance
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = ColorFormatter(
            '%(asctime)s │ %(levelname)s │ %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to: {log_file}")

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.

    Returns:
        Logger instance (creates default if not set up)
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


class PhaseLogger:
    """Context manager for logging pipeline phases."""

    def __init__(self, phase_name: str, logger: Optional[logging.Logger] = None):
        self.phase_name = phase_name
        self.logger = logger or get_logger()
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting: {self.phase_name}")
        self.logger.info(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(f"Completed: {self.phase_name} ({elapsed:.1f}s)")
        else:
            self.logger.error(f"Failed: {self.phase_name} ({elapsed:.1f}s)")
            self.logger.error(f"Error: {exc_val}")

        return False  # Don't suppress exceptions

    def step(self, message: str) -> None:
        """Log a step within the phase."""
        self.logger.info(f"  → {message}")

    def substep(self, message: str) -> None:
        """Log a substep within the phase."""
        self.logger.info(f"    • {message}")

    def warn(self, message: str) -> None:
        """Log a warning."""
        self.logger.warning(f"  ⚠ {message}")

    def error(self, message: str) -> None:
        """Log an error."""
        self.logger.error(f"  ✗ {message}")

    def success(self, message: str) -> None:
        """Log a success message."""
        self.logger.info(f"  ✓ {message}")

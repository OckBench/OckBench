"""Logging configuration and utilities."""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ockbench",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _sanitize_name(name: str) -> str:
    """Sanitize name for use in filenames."""
    return name.replace('/', '_').replace('\\', '_')


def _get_filename(dataset_name: str, model_name: str, ext: str, timestamp: Optional[str] = None) -> str:
    """Generate standardized filename for experiment artifacts."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_name(dataset_name)}_{_sanitize_name(model_name)}_{timestamp}.{ext}"


def get_experiment_filename(dataset_name: str, model_name: str, timestamp: Optional[str] = None) -> str:
    """Generate filename for experiment results (.json)."""
    return _get_filename(dataset_name, model_name, "json", timestamp)


def get_log_filename(dataset_name: str, model_name: str, timestamp: Optional[str] = None) -> str:
    """Generate filename for log files (.log)."""
    return _get_filename(dataset_name, model_name, "log", timestamp)

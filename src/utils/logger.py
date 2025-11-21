"""
Logging configuration and utilities for OckBench.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "ockbench",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console: Whether to log to console
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_filename(dataset_name: str, model_name: str, timestamp: Optional[str] = None) -> str:
    """
    Generate standardized filename for experiment results.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        timestamp: Timestamp string (optional, will generate if not provided)
    
    Returns:
        str: Filename in format: {dataset}_{model}_{timestamp}.json
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize names for filenames
    dataset_clean = dataset_name.replace('/', '_').replace('\\', '_')
    model_clean = model_name.replace('/', '_').replace('\\', '_')
    
    return f"{dataset_clean}_{model_clean}_{timestamp}.json"


def get_log_filename(dataset_name: str, model_name: str, timestamp: Optional[str] = None) -> str:
    """
    Generate standardized filename for log files.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        timestamp: Timestamp string (optional, will generate if not provided)
    
    Returns:
        str: Filename in format: {dataset}_{model}_{timestamp}.log
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize names for filenames
    dataset_clean = dataset_name.replace('/', '_').replace('\\', '_')
    model_clean = model_name.replace('/', '_').replace('\\', '_')
    
    return f"{dataset_clean}_{model_clean}_{timestamp}.log"


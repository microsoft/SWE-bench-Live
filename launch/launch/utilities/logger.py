"""
Logging utilities for launch operations with file and console output.
"""
import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(instance_id: str, log_file: Path, printing: bool = True) -> logging.Logger:
    """
    Setup logger with file and optional console output for an instance.
    
    Args:
        instance_id (str): Unique identifier for the logger instance
        log_file (Path): Path to the log file
        printing (bool): Whether to enable console output with rich formatting
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(instance_id)
    logger.setLevel(logging.INFO)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # add console handler
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # logger.addHandler(ch)
    # add rich handler
    if printing:
        rh = RichHandler()
        rh.setLevel(logging.INFO)
        logger.addHandler(rh)
    return logger


def clean_logger(logger: logging.Logger | str) -> None:
    """
    Clean up logger handlers to prevent resource leaks.
    
    Args:
        logger (logging.Logger | str): Logger instance or instance ID string
    """
    if isinstance(logger, str):  # get logger by instance id
        logger = logging.getLogger(logger)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

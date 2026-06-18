# -------------------------------------------------------------------
# logger_setup.py
# Configures application-wide logging based on settings.
# -------------------------------------------------------------------

import logging
import os
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]) -> logging.Logger:
    """Configure and return the application logger.

    Sets up logging with the specified level, format, and optional file output.
    Creates log directories if they do not exist.

    Args:
        logging_config: Dictionary containing logging configuration values
                       (level, format, file).

    Returns:
        Configured root logger instance.
    """
    # Extract configuration values with sensible defaults
    log_level = logging_config.get("level", "INFO").upper()
    log_format = logging_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = logging_config.get("file", None)

    # Create the root logger for the application
    root_logger = logging.getLogger("llm_cache")
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Create a formatter with the configured format string
    formatter = logging.Formatter(log_format)

    # Add console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if a log file path is configured
    if log_file:
        # Create the log directory if it doesn't exist
        log_directory = os.path.dirname(log_file)
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)

        # Create file handler for persistent log storage
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info("Logging configured at level: %s", log_level)
    return root_logger

"""
Centralized logging configuration for the AI Interview System.

This module provides a unified logging setup that can be used across all files
in the src folder. It integrates with the application's config.yaml for settings
and provides both console and file logging with proper formatting.

Usage:
    from src.utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("This is an info message")
    logger.error("This is an error message")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Global flag to track if logging has been initialized
_logging_initialized = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "./logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_colors: bool = True
) -> logging.Logger:
    """
    Configure logging for the application.
    
    This function sets up both console and file logging with appropriate
    formatters and handlers. It should be called once at application startup.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
        use_colors: Whether to use colored output in console (default: True)
    
    Returns:
        The configured root logger
    """
    global _logging_initialized
    
    # Avoid re-initializing if already done
    if _logging_initialized:
        return logging.getLogger()
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ) if use_colors else logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(log_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress overly verbose third-party loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    _logging_initialized = True
    
    root_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    This is the recommended way to get a logger in any module. It ensures
    that logging is properly initialized and returns a logger with the
    appropriate name.
    
    Args:
        name: Name for the logger (typically __name__ of the calling module)
    
    Returns:
        A configured logger instance
    
    Example:
        >>> from src.utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Initialize logging if not already done
    if not _logging_initialized:
        # Try to load settings from config
        try:
            from src.utils.config import get_settings
            settings = get_settings()
            if settings and settings.log_level:
                setup_logging(
                    log_level=settings.log_level
                )
            else:
                setup_logging()
        except Exception:
            # Fallback to default settings if config loading fails
            setup_logging()
    
    return logging.getLogger(name)


# Initialize logging on module import with config settings
try:
    from src.utils.config import get_settings
    settings = get_settings()
    if settings and settings.log_level:
        setup_logging(
            log_level=settings.log_level,
        )
    else:
        setup_logging()
except Exception as e:
    # Fallback to default settings if config loading fails
    setup_logging()
    logging.getLogger(__name__).warning(
        f"Could not load logging config from settings, using defaults: {e}"
    )
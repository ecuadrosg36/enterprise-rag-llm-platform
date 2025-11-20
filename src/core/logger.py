"""
Structured JSON logging setup with correlation ID support.

Provides production-grade logging with:
- JSON formatting for log aggregation
- Rotating file handlers
- Console output with optional colors
- Correlation ID injection for request tracing
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import contextvars

# Correlation ID context variable for request tracing
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)

        # Add correlation ID to message if available
        correlation_id = correlation_id_var.get()
        correlation_str = f" [{correlation_id}]" if correlation_id else ""

        formatted = (
            f"{color}[{record.levelname}]{self.RESET} "
            f"{record.name}{correlation_str} - "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_json: bool = True,
    use_console: bool = True,
    colorize_console: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up structured logger with file and console handlers.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = no file logging)
        use_json: Use JSON formatting for file logs
        use_console: Enable console logging
        colorize_console: Use colored output for console
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # File handler with JSON formatting
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )

        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        logger.addHandler(file_handler)

    # Console handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)

        if colorize_console and sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter("%(levelname)s - %(name)s - %(message)s")
            )

        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context (e.g., HTTP request)."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get correlation ID from current context."""
    return correlation_id_var.get()


def clear_correlation_id():
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes extra fields."""

    def process(self, msg, kwargs):
        """Add extra fields to log record."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Add correlation ID if not already present
        if "correlation_id" not in kwargs["extra"]:
            correlation_id = get_correlation_id()
            if correlation_id:
                kwargs["extra"]["correlation_id"] = correlation_id

        return msg, kwargs

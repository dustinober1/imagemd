"""
Logging configuration and utilities for VisionPDF.

This module provides centralized logging configuration with support for
different log levels, file output, and structured logging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from ..config.settings import VisionPDFConfig, LogLevel


class VisionPDFLogger:
    """
    Centralized logging configuration for VisionPDF.

    This class provides consistent logging configuration across the package
    with support for file rotation, structured logging, and different output formats.
    """

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the logging configuration.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_root_logger()
        self._setup_package_logger()

    def _setup_root_logger(self) -> None:
        """Configure the root logger for the application."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.logging.level.value))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatters
        formatter = self._create_formatter()

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.config.logging.level.value))
        root_logger.addHandler(console_handler)

        # Add file handler if configured
        if self.config.logging.file:
            file_handler = self._create_file_handler()
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, self.config.logging.level.value))
            root_logger.addHandler(file_handler)

    def _setup_package_logger(self) -> None:
        """Configure the main package logger."""
        package_logger = logging.getLogger("vision_pdf")
        package_logger.setLevel(getattr(logging, self.config.logging.level.value))

        # Store reference for easy access
        self.loggers["vision_pdf"] = package_logger

    def _create_formatter(self) -> logging.Formatter:
        """
        Create a log formatter based on configuration.

        Returns:
            Configured logging formatter
        """
        format_string = self.config.logging.format

        # Add structured logging elements if needed
        if self._should_use_structured_logging():
            format_string = self._add_structured_elements(format_string)

        return logging.Formatter(
            fmt=format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _create_file_handler(self) -> logging.Handler:
        """
        Create a file handler with rotation if configured.

        Returns:
            Configured file handler
        """
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.config.logging.max_size > 0 and self.config.logging.backup_count > 0:
            # Use rotating file handler
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.logging.max_size,
                backupCount=self.config.logging.backup_count,
                encoding='utf-8'
            )
        else:
            # Use basic file handler
            handler = logging.FileHandler(
                log_file,
                encoding='utf-8'
            )

        return handler

    def _should_use_structured_logging(self) -> bool:
        """
        Determine if structured logging should be used.

        Returns:
            True if structured logging is appropriate
        """
        # Use structured logging for higher log levels or when file logging is enabled
        return (
            self.config.logging.level in [LogLevel.DEBUG, LogLevel.INFO] or
            self.config.logging.file is not None
        )

    def _add_structured_elements(self, format_string: str) -> str:
        """
        Add structured elements to the log format.

        Args:
            format_string: Base format string

        Returns:
            Enhanced format string with structured elements
        """
        # Add module, function, and line number for better debugging
        if "%(module)s" not in format_string:
            format_string = format_string.replace(
                "%(name)s",
                "%(name)s[%(module)s:%(funcName)s:%(lineno)d]"
            )

        return format_string

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific module.

        Args:
            name: Logger name (usually __name__)

        Returns:
            Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger

        return self.loggers[name]

    def set_level(self, level: LogLevel) -> None:
        """
        Change the logging level for all loggers.

        Args:
            level: New logging level
        """
        log_level = getattr(logging, level.value)

        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(log_level)

        # Update package loggers
        for logger in self.loggers.values():
            logger.setLevel(log_level)

    def log_performance(
        self,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance information.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Additional performance metadata
        """
        logger = self.get_logger("vision_pdf.performance")

        message = f"Operation '{operation}' completed in {duration:.3f}s"

        if metadata:
            message += f" | Metadata: {json.dumps(metadata)}"

        logger.info(message)

    def log_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error information with context.

        Args:
            error: Exception that occurred
            operation: Operation that failed
            context: Additional context information
        """
        logger = self.get_logger("vision_pdf.error")

        message = f"Error in operation '{operation}': {str(error)}"

        if context:
            message += f" | Context: {json.dumps(context)}"

        logger.error(message, exc_info=True)


class StructuredLogger:
    """
    Logger for structured logging with JSON output.

    This class provides structured logging capabilities for better
    log analysis and monitoring in production environments.
    """

    def __init__(self, name: str, config: VisionPDFConfig):
        """
        Initialize the structured logger.

        Args:
            name: Logger name
            config: VisionPDF configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)

    def log_event(
        self,
        event_type: str,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> None:
        """
        Log a structured event.

        Args:
            event_type: Type of event
            level: Log level
            message: Log message
            **kwargs: Additional event data
        """
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "logger": self.name,
            "message": message,
            **kwargs
        }

        # Convert to JSON string for logging
        json_message = json.dumps(event_data, default=str)

        log_method = getattr(self.logger, level.value.lower())
        log_method(json_message)

    def log_request(
        self,
        request_id: str,
        operation: str,
        status: str,
        duration: float,
        **kwargs
    ) -> None:
        """
        Log a request/response event.

        Args:
            request_id: Unique request identifier
            operation: Operation being performed
            status: Request status (success, error, timeout)
            duration: Request duration in seconds
            **kwargs: Additional request data
        """
        self.log_event(
            event_type="request",
            level=LogLevel.INFO,
            message=f"Request {request_id}: {operation} - {status}",
            request_id=request_id,
            operation=operation,
            status=status,
            duration=duration,
            **kwargs
        )

    def log_system_event(
        self,
        event_type: str,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> None:
        """
        Log a system-level event.

        Args:
            event_type: Type of system event
            level: Log level
            message: Event message
            **kwargs: Additional system data
        """
        self.log_event(
            event_type=f"system_{event_type}",
            level=level,
            message=message,
            **kwargs
        )


# Global logger instance
_logger_instance: Optional[VisionPDFLogger] = None


def setup_logging(config: VisionPDFConfig) -> VisionPDFLogger:
    """
    Set up logging for the VisionPDF package.

    Args:
        config: VisionPDF configuration

    Returns:
        Configured logger instance
    """
    global _logger_instance
    _logger_instance = VisionPDFLogger(config)
    return _logger_instance


def setup_cli_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Simple logging setup for CLI usage.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    import logging
    import sys

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if _logger_instance is None:
        # Create default logger if not configured
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

    return _logger_instance.get_logger(name)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    if _logger_instance is None:
        raise RuntimeError("Logging not configured. Call setup_logging() first.")

    return StructuredLogger(name, _logger_instance.config)


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Log performance information.

    Args:
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metadata
    """
    if _logger_instance:
        _logger_instance.log_performance(operation, duration, kwargs)


def log_error(error: Exception, operation: str, **kwargs) -> None:
    """
    Log error information.

    Args:
        error: Exception that occurred
        operation: Operation that failed
        **kwargs: Additional context
    """
    if _logger_instance:
        _logger_instance.log_error(error, operation, kwargs)
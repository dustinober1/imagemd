"""
Utility modules for VisionPDF.

This module contains utility functions and classes for logging,
error handling, caching, image processing, and other common tasks.
"""

from .logging_config import (
    VisionPDFLogger,
    StructuredLogger,
    setup_logging,
    get_logger,
    get_structured_logger,
    log_performance,
    log_error
)

from .exceptions import (
    VisionPDFError,
    ConfigurationError,
    PDFProcessingError,
    PDFRenderingError,
    PDFAnalysisError,
    TextExtractionError,
    OCRError,
    BackendError,
    BackendConnectionError,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    MarkdownGenerationError,
    ValidationError,
    CacheError,
    TimeoutError,
    ResourceError,
    DependencyError,
    AuthenticationError,
    RateLimitError,
    ErrorCodes,
    wrap_exception,
    is_retriable_error,
    get_error_severity
)

__all__ = [
    # Logging
    "VisionPDFLogger",
    "StructuredLogger",
    "setup_logging",
    "get_logger",
    "get_structured_logger",
    "log_performance",
    "log_error",

    # Exceptions
    "VisionPDFError",
    "ConfigurationError",
    "PDFProcessingError",
    "PDFRenderingError",
    "PDFAnalysisError",
    "TextExtractionError",
    "OCRError",
    "BackendError",
    "BackendConnectionError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InferenceError",
    "MarkdownGenerationError",
    "ValidationError",
    "CacheError",
    "TimeoutError",
    "ResourceError",
    "DependencyError",
    "AuthenticationError",
    "RateLimitError",
    "ErrorCodes",
    "wrap_exception",
    "is_retriable_error",
    "get_error_severity"
]
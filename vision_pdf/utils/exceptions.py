"""
Custom exceptions for VisionPDF.

This module defines custom exception classes for different types of errors
that can occur during PDF processing and vision language model operations.
"""


class VisionPDFError(Exception):
    """
    Base exception class for all VisionPDF errors.

    This is the parent class for all custom exceptions in the VisionPDF package.
    """

    def __init__(self, message: str, error_code: str = None, context: dict = None):
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Optional error code for machine-readable identification
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message

    def to_dict(self) -> dict:
        """
        Convert exception to dictionary for serialization.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class ConfigurationError(VisionPDFError):
    """
    Exception raised for configuration-related errors.
    """

    def __init__(self, message: str, config_key: str = None, config_value: str = None):
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value:
            context["config_value"] = config_value

        super().__init__(message, "CONFIG_ERROR", context)
        self.config_key = config_key
        self.config_value = config_value


class PDFProcessingError(VisionPDFError):
    """
    Exception raised for errors during PDF processing.
    """

    def __init__(self, message: str, pdf_path: str = None, page_number: int = None, operation: str = None):
        context = {}
        if pdf_path:
            context["pdf_path"] = pdf_path
        if page_number is not None:
            context["page_number"] = page_number
        if operation:
            context["operation"] = operation

        super().__init__(message, "PDF_PROCESSING_ERROR", context)
        self.pdf_path = pdf_path
        self.page_number = page_number
        self.operation = operation


class PDFRenderingError(PDFProcessingError):
    """
    Exception raised for errors during PDF rendering.
    """

    def __init__(self, message: str, pdf_path: str = None, page_number: int = None, dpi: int = None):
        context = {}
        if pdf_path:
            context["pdf_path"] = pdf_path
        if page_number is not None:
            context["page_number"] = page_number
        if dpi:
            context["dpi"] = dpi

        super().__init__(message, pdf_path, page_number, "rendering")
        self.dpi = dpi


class PDFAnalysisError(PDFProcessingError):
    """
    Exception raised for errors during PDF analysis.
    """

    def __init__(self, message: str, pdf_path: str = None, page_number: int = None):
        super().__init__(message, pdf_path, page_number, "analysis")


class TextExtractionError(PDFProcessingError):
    """
    Exception raised for errors during text extraction.
    """

    def __init__(self, message: str, pdf_path: str = None, page_number: int = None, method: str = None):
        context = {}
        if method:
            context["extraction_method"] = method

        super().__init__(message, pdf_path, page_number, "text_extraction")
        self.method = method
        if method:
            self.context["extraction_method"] = method


class OCRError(PDFProcessingError):
    """
    Exception raised for errors during OCR processing.
    """

    def __init__(self, message: str, pdf_path: str = None, page_number: int = None, language: str = None):
        context = {}
        if language:
            context["language"] = language

        super().__init__(message, pdf_path, page_number, "ocr")
        self.language = language
        if language:
            self.context["language"] = language


class BackendError(VisionPDFError):
    """
    Exception raised for VLM backend errors.
    """

    def __init__(self, message: str, backend_type: str = None, model_name: str = None):
        context = {}
        if backend_type:
            context["backend_type"] = backend_type
        if model_name:
            context["model_name"] = model_name

        super().__init__(message, "BACKEND_ERROR", context)
        self.backend_type = backend_type
        self.model_name = model_name


class BackendConnectionError(BackendError):
    """
    Exception raised for backend connection errors.
    """

    def __init__(self, message: str, backend_type: str = None, endpoint: str = None):
        context = {}
        if endpoint:
            context["endpoint"] = endpoint

        super().__init__(message, backend_type)
        self.endpoint = endpoint
        if endpoint:
            self.context["endpoint"] = endpoint


class ModelNotFoundError(BackendError):
    """
    Exception raised when a requested model is not found.
    """

    def __init__(self, model_name: str, backend_type: str = None):
        message = f"Model '{model_name}' not found"
        super().__init__(message, backend_type, model_name)


class ModelLoadError(BackendError):
    """
    Exception raised when a model fails to load.
    """

    def __init__(self, model_name: str, backend_type: str = None, load_time: float = None):
        context = {}
        if load_time:
            context["load_time"] = load_time

        message = f"Failed to load model '{model_name}'"
        super().__init__(message, backend_type, model_name)
        self.load_time = load_time
        if load_time:
            self.context["load_time"] = load_time


class InferenceError(BackendError):
    """
    Exception raised during model inference.
    """

    def __init__(self, message: str, model_name: str = None, input_size: int = None, processing_time: float = None):
        context = {}
        if input_size:
            context["input_size"] = input_size
        if processing_time:
            context["processing_time"] = processing_time

        super().__init__(message, model_name=model_name)
        self.input_size = input_size
        self.processing_time = processing_time
        if input_size:
            self.context["input_size"] = input_size
        if processing_time:
            self.context["processing_time"] = processing_time


class MarkdownGenerationError(VisionPDFError):
    """
    Exception raised during markdown generation.
    """

    def __init__(self, message: str, page_number: int = None, content_type: str = None):
        context = {}
        if page_number is not None:
            context["page_number"] = page_number
        if content_type:
            context["content_type"] = content_type

        super().__init__(message, "MARKDOWN_GENERATION_ERROR", context)
        self.page_number = page_number
        self.content_type = content_type


class ValidationError(VisionPDFError):
    """
    Exception raised for validation errors.
    """

    def __init__(self, message: str, field: str = None, value: str = None):
        context = {}
        if field:
            context["field"] = field
        if value:
            context["value"] = value

        super().__init__(message, "VALIDATION_ERROR", context)
        self.field = field
        self.value = value


class CacheError(VisionPDFError):
    """
    Exception raised for cache-related errors.
    """

    def __init__(self, message: str, cache_type: str = None, cache_key: str = None):
        context = {}
        if cache_type:
            context["cache_type"] = cache_type
        if cache_key:
            context["cache_key"] = cache_key

        super().__init__(message, "CACHE_ERROR", context)
        self.cache_type = cache_type
        self.cache_key = cache_key


class TimeoutError(VisionPDFError):
    """
    Exception raised when operations timeout.
    """

    def __init__(self, message: str, operation: str = None, timeout: float = None, elapsed_time: float = None):
        context = {}
        if operation:
            context["operation"] = operation
        if timeout:
            context["timeout"] = timeout
        if elapsed_time:
            context["elapsed_time"] = elapsed_time

        super().__init__(message, "TIMEOUT_ERROR", context)
        self.operation = operation
        self.timeout = timeout
        self.elapsed_time = elapsed_time


class ResourceError(VisionPDFError):
    """
    Exception raised for resource-related errors (memory, disk space, etc.).
    """

    def __init__(self, message: str, resource_type: str = None, usage: float = None, limit: float = None):
        context = {}
        if resource_type:
            context["resource_type"] = resource_type
        if usage:
            context["usage"] = usage
        if limit:
            context["limit"] = limit

        super().__init__(message, "RESOURCE_ERROR", context)
        self.resource_type = resource_type
        self.usage = usage
        self.limit = limit


class DependencyError(VisionPDFError):
    """
    Exception raised when required dependencies are missing.
    """

    def __init__(self, message: str, dependency: str = None, version: str = None):
        context = {}
        if dependency:
            context["dependency"] = dependency
        if version:
            context["version"] = version

        super().__init__(message, "DEPENDENCY_ERROR", context)
        self.dependency = dependency
        self.version = version


class AuthenticationError(BackendError):
    """
    Exception raised for authentication errors with backend services.
    """

    def __init__(self, message: str, backend_type: str = None, endpoint: str = None):
        super().__init__(message, backend_type)
        self.endpoint = endpoint
        if endpoint:
            self.context["endpoint"] = endpoint


class RateLimitError(BackendError):
    """
    Exception raised when rate limits are exceeded.
    """

    def __init__(self, message: str, backend_type: str = None, retry_after: int = None):
        context = {}
        if retry_after:
            context["retry_after"] = retry_after

        super().__init__(message, backend_type)
        self.retry_after = retry_after
        if retry_after:
            self.context["retry_after"] = retry_after


# Error code constants for machine-readable error identification
class ErrorCodes:
    """Constants for error codes."""
    CONFIG_ERROR = "CONFIG_ERROR"
    PDF_PROCESSING_ERROR = "PDF_PROCESSING_ERROR"
    PDF_RENDERING_ERROR = "PDF_RENDERING_ERROR"
    TEXT_EXTRACTION_ERROR = "TEXT_EXTRACTION_ERROR"
    OCR_ERROR = "OCR_ERROR"
    BACKEND_ERROR = "BACKEND_ERROR"
    BACKEND_CONNECTION_ERROR = "BACKEND_CONNECTION_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    MARKDOWN_GENERATION_ERROR = "MARKDOWN_GENERATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"


def wrap_exception(
    exception: Exception,
    error_class: type = VisionPDFError,
    message: str = None,
    **context
) -> VisionPDFError:
    """
    Wrap a generic exception in a VisionPDF exception.

    Args:
        exception: Original exception to wrap
        error_class: VisionPDF exception class to use
        message: Custom error message
        **context: Additional context information

    Returns:
        Wrapped VisionPDF exception
    """
    if message is None:
        message = str(exception)

    if isinstance(exception, VisionPDFError):
        # Already a VisionPDF exception, just add context
        if context:
            exception.context.update(context)
        return exception

    # Create new VisionPDF exception
    return error_class(message, **context)


def is_retriable_error(error: Exception) -> bool:
    """
    Determine if an error is retriable.

    Args:
        error: Exception to check

    Returns:
        True if the error can be retried
    """
    retriable_error_types = (
        BackendConnectionError,
        TimeoutError,
        RateLimitError,
        ModelLoadError
    )

    return isinstance(error, retriable_error_types)


def get_error_severity(error: Exception) -> str:
    """
    Get the severity level of an error.

    Args:
        error: Exception to evaluate

    Returns:
        Severity level: 'low', 'medium', 'high', 'critical'
    """
    critical_errors = (
        ConfigurationError,
        DependencyError,
        AuthenticationError
    )

    high_severity_errors = (
        BackendConnectionError,
        ModelNotFoundError,
        ResourceError
    )

    medium_severity_errors = (
        PDFProcessingError,
        InferenceError,
        MarkdownGenerationError
    )

    if isinstance(error, critical_errors):
        return "critical"
    elif isinstance(error, high_severity_errors):
        return "high"
    elif isinstance(error, medium_severity_errors):
        return "medium"
    else:
        return "low"
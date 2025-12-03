"""
Pydantic models for the VisionPDF REST API.

This module defines the request and response models used by the FastAPI endpoints.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator

from ..backends.base import BackendType, ProcessingMode


class JobStatus(str, Enum):
    """Enumeration for job status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversionConfig(BaseModel):
    """Configuration for PDF conversion process."""
    backend_type: BackendType = Field(
        default=BackendType.OLLAMA,
        description="VLM backend to use for conversion"
    )
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.HYBRID,
        description="Processing mode for conversion"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model name to use (overrides default)"
    )

    # Format preservation options
    preserve_tables: bool = Field(
        default=True,
        description="Preserve and format tables"
    )
    preserve_math: bool = Field(
        default=True,
        description="Preserve and convert mathematical formulas"
    )
    preserve_code: bool = Field(
        default=True,
        description="Preserve and syntax-highlight code blocks"
    )
    preserve_images: bool = Field(
        default=True,
        description="Include image references in output"
    )

    # Processing options
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing of pages"
    )
    ocr_fallback: bool = Field(
        default=True,
        description="Use OCR as fallback for text extraction"
    )
    dpi: int = Field(
        default=200,
        ge=72,
        le=600,
        description="DPI for PDF rendering"
    )
    max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of pages to process"
    )

    # Quality settings
    quality: str = Field(
        default="balanced",
        regex="^(fast|balanced|high)$",
        description="Processing quality preset"
    )

    @validator('model_name')
    def validate_model_name(cls, v, values):
        """Validate model name based on backend type."""
        if v is not None:
            if not isinstance(v, str) or len(v.strip()) == 0:
                raise ValueError("Model name must be a non-empty string")
        return v


class ConversionRequest(BaseModel):
    """Request model for PDF conversion job."""
    config: ConversionConfig = Field(
        default_factory=ConversionConfig,
        description="Conversion configuration"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Job priority (1=lowest, 10=highest)"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to call when conversion completes"
    )
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Secret for webhook authentication"
    )

    @validator('callback_url')
    def validate_callback_url(cls, v):
        """Validate callback URL format."""
        if v is not None:
            if not v.startswith(('http://', 'https://')):
                raise ValueError("Callback URL must start with http:// or https://")
        return v


class ConversionProgress(BaseModel):
    """Model for conversion progress updates."""
    job_id: str
    status: JobStatus
    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Progress percentage (0.0 to 1.0)"
    )
    message: Optional[str] = None
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    processing_time: Optional[float] = None
    estimated_time_remaining: Optional[float] = None


class ConversionResult(BaseModel):
    """Model for conversion results."""
    job_id: str
    status: JobStatus
    markdown_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Statistics
    pages_processed: int = 0
    processing_time: float = 0.0
    tables_detected: int = 0
    math_formulas_detected: int = 0
    code_blocks_detected: int = 0
    images_extracted: int = 0

    # Quality metrics
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence in conversion quality"
    )

    # File information
    original_filename: Optional[str] = None
    original_size: Optional[int] = None
    markdown_size: Optional[int] = None


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str = Field(description="Error message")
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Model for health check responses."""
    status: str = Field(description="Service health status")
    message: str
    version: str
    uptime: float
    system_info: Dict[str, Any]

    # Backend information
    available_backends: List[str]
    backend_status: Dict[str, Dict[str, Any]]

    # Service statistics
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_conversions: int = 0


class ModelInfo(BaseModel):
    """Model information for available VLM models."""
    name: str
    backend: BackendType
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    max_context_length: Optional[int] = None
    supported_formats: List[str] = Field(default_factory=list)
    is_available: bool = True


class ModelsResponse(BaseModel):
    """Response model for available models endpoint."""
    backend_type: BackendType
    models: List[ModelInfo]
    count: int
    default_model: Optional[str] = None


class JobInfo(BaseModel):
    """Information about a conversion job."""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Job details
    config: ConversionConfig
    original_filename: Optional[str] = None
    file_size: Optional[int] = None

    # Progress information
    message: Optional[str] = None
    current_page: Optional[int] = None
    total_pages: Optional[int] = None

    # Results (if completed)
    result: Optional[ConversionResult] = None

    # Error information (if failed)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class JobsListResponse(BaseModel):
    """Response model for jobs list endpoint."""
    jobs: List[JobInfo]
    count: int
    page: int = 1
    page_size: int = 10
    total_count: Optional[int] = None


class BatchConversionRequest(BaseModel):
    """Request model for batch PDF conversion."""
    files: List[str] = Field(..., min_items=1, max_items=50)
    config: ConversionConfig = Field(default_factory=ConversionConfig)
    output_format: str = Field(default="markdown", regex="^(markdown|html|text)$")

    # Batch options
    parallel_jobs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of parallel conversion jobs"
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop batch on first error"
    )


class BatchConversionResponse(BaseModel):
    """Response model for batch conversion."""
    batch_id: str
    job_ids: List[str]
    status: JobStatus
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    progress: float = Field(ge=0.0, le=1.0)


class WebhookPayload(BaseModel):
    """Model for webhook callback payloads."""
    event: str
    job_id: str
    status: JobStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None


class ApiConfiguration(BaseModel):
    """Configuration for the API server."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)

    # Security settings
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_key_required: bool = Field(default=False)
    max_file_size: int = Field(default=50 * 1024 * 1024)  # 50MB

    # Performance settings
    max_concurrent_jobs: int = Field(default=10, ge=1, le=100)
    job_timeout: int = Field(default=600, ge=30)  # 10 minutes
    cleanup_interval: int = Field(default=300, ge=60)  # 5 minutes

    # Storage settings
    temp_dir: str = Field(default="/tmp")
    max_job_age: int = Field(default=86400, ge=3600)  # 24 hours


class MetricsResponse(BaseModel):
    """Response model for API metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)

    # Job statistics
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int

    # Performance metrics
    average_processing_time: float
    average_file_size: float
    average_pages_processed: float

    # System metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None

    # Backend metrics
    backend_metrics: Dict[str, Dict[str, Any]]


class ValidationRequest(BaseModel):
    """Request model for file validation."""
    filename: str
    file_size: int
    file_type: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for file validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    file_info: Dict[str, Any] = Field(default_factory=dict)


# Utility functions for creating responses
def create_error_response(message: str, error_code: Optional[str] = None, details: Optional[Dict] = None) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=message,
        error_code=error_code,
        details=details
    )


def create_success_response(message: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Create a standardized success response."""
    response = {"message": message, "status": "success"}
    if data:
        response.update(data)
    return response
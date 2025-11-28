"""
Base classes and interfaces for vision language model backends.

This module defines the abstract interface that all VLM backends must implement,
along with common enums and data structures used across backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import pathlib


class BackendType(Enum):
    """Supported vision language model backend types."""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    CUSTOM_API = "custom_api"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class ProcessingMode(Enum):
    """Processing modes for PDF to markdown conversion."""
    VISION_ONLY = "vision_only"  # Convert PDF pages to images and use VLM for complete analysis
    HYBRID = "hybrid"           # Extract text via OCR/PyPDF and use VLM for layout/formatting
    TEXT_ONLY = "text_only"     # Traditional text extraction only (no vision processing)


class ModelStatus(Enum):
    """Status of vision language models."""
    AVAILABLE = "available"
    LOADING = "loading"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a vision language model."""
    name: str
    backend_type: BackendType
    status: ModelStatus
    description: Optional[str] = None
    supported_modes: List[ProcessingMode] = None
    max_image_size: Optional[int] = None
    requires_api_key: bool = False

    def __post_init__(self):
        """Initialize default values."""
        if self.supported_modes is None:
            self.supported_modes = [ProcessingMode.VISION_ONLY, ProcessingMode.HYBRID]


@dataclass
class ProcessingRequest:
    """Request for processing a PDF page with a VLM."""
    image_path: pathlib.Path
    text_content: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.VISION_ONLY
    prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.prompt is None:
            self.prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default prompt based on processing mode."""
        if self.processing_mode == ProcessingMode.VISION_ONLY:
            return (
                "Convert this PDF page to well-formatted markdown. "
                "Preserve the layout, tables, mathematical formulas, and code blocks. "
                "Use proper markdown formatting for headings, lists, and other elements."
            )
        elif self.processing_mode == ProcessingMode.HYBRID:
            return (
                "I have extracted the following text from this PDF page: {text}\n\n"
                "Please analyze the page image and convert it to well-formatted markdown. "
                "Use the extracted text as reference but preserve the exact visual layout. "
                "Format tables properly, convert mathematical formulas to LaTeX, "
                "and identify code blocks with appropriate syntax highlighting."
            )
        else:
            return "Format the provided text into well-structured markdown."


@dataclass
class ProcessingResponse:
    """Response from processing a PDF page with a VLM."""
    markdown: str
    confidence: float
    processing_time: float
    model_used: str
    elements_detected: List[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.elements_detected is None:
            self.elements_detected = []
        if self.metadata is None:
            self.metadata = {}
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")


class VLMBackend(ABC):
    """
    Abstract base class for all vision language model backends.

    This class defines the interface that all VLM backends must implement.
    It provides methods for model management, processing, and configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VLM backend with configuration.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self._model_info: Optional[ModelInfo] = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and establish connection to the model.

        This method should be called before any processing requests.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up resources and close connections.

        This method should be called when the backend is no longer needed.
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available models for this backend.

        Returns:
            List of ModelInfo objects describing available models
        """
        pass

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connection to the backend.

        Returns:
            True if connection is successful, False otherwise
        """
        pass

    @abstractmethod
    async def process_page(
        self,
        request: ProcessingRequest,
        progress_callback: Optional[callable] = None
    ) -> ProcessingResponse:
        """
        Process a single PDF page using the vision language model.

        Args:
            request: ProcessingRequest containing the page and processing parameters
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResponse with the generated markdown and metadata
        """
        pass

    async def process_batch(
        self,
        requests: List[ProcessingRequest],
        progress_callback: Optional[callable] = None,
        max_concurrent: int = 4
    ) -> List[ProcessingResponse]:
        """
        Process multiple pages in parallel.

        Args:
            requests: List of ProcessingRequest objects
            progress_callback: Optional callback for progress updates
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of ProcessingResponse objects
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(req: ProcessingRequest, index: int) -> tuple[int, ProcessingResponse]:
            async with semaphore:
                response = await self.process_page(req, progress_callback)
                return index, response

        # Create tasks for all requests
        tasks = [process_single(req, i) for i, req in enumerate(requests)]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by original order
        responses = [None] * len(requests)
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions by creating error responses
                error_response = ProcessingResponse(
                    markdown="",
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="unknown",
                    error_message=str(result)
                )
                # Find the index (this is a simplified approach)
                for i, _ in enumerate(responses):
                    if responses[i] is None:
                        responses[i] = error_response
                        break
            else:
                index, response = result
                responses[index] = response

        return responses

    @property
    def model_info(self) -> Optional[ModelInfo]:
        """Get information about the currently loaded model."""
        return self._model_info

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return self.config.get("backend_type", BackendType.CUSTOM_API)

    def validate_config(self) -> bool:
        """
        Validate the backend configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = self._get_required_config_keys()
        return all(key in self.config for key in required_keys)

    @abstractmethod
    def _get_required_config_keys(self) -> List[str]:
        """
        Get list of required configuration keys for this backend.

        Returns:
            List of required configuration key names
        """
        pass

    def __repr__(self) -> str:
        """String representation of the backend."""
        return f"{self.__class__.__name__}(backend_type={self.backend_type.value})"
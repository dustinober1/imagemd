"""
Vision language model backends for VisionPDF.

This module provides implementations of various VLM backends including
Ollama, llama.cpp, and custom API endpoints.
"""

from .base import (
    VLMBackend,
    BackendType,
    ProcessingMode,
    ModelInfo,
    ModelStatus,
    ProcessingRequest,
    ProcessingResponse
)

# Import specific backend implementations when they are created
# from .ollama import OllamaBackend
# from .llama_cpp import LlamaCppBackend
# from .custom import CustomAPIBackend

__all__ = [
    "VLMBackend",
    "BackendType",
    "ProcessingMode",
    "ModelInfo",
    "ModelStatus",
    "ProcessingRequest",
    "ProcessingResponse",
    # Add backend implementations when created
    # "OllamaBackend",
    # "LlamaCppBackend",
    # "CustomAPIBackend"
]
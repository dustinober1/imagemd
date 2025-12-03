"""
Model information and management routes.

This module provides endpoints for listing and managing VLM models.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...core.processor import VisionPDF
from ...config.settings import VisionPDFConfig
from ...backends.base import BackendType
from ..models import ModelInfo, ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])


async def get_models_for_backend(backend_type: BackendType) -> List[ModelInfo]:
    """Get available models for a specific backend."""
    try:
        config = VisionPDFConfig()
        converter = VisionPDF(
            config=config,
            backend_type=backend_type
        )

        # Get available models
        model_names = await converter.get_available_models()
        await converter.close()

        # Create model info objects
        models = []
        for model_name in model_names:
            model_info = ModelInfo(
                name=model_name,
                backend=backend_type,
                description=f"Model available on {backend_type.value} backend",
                capabilities=["vision", "text-generation"],  # Basic capabilities
                supported_formats=["pdf", "image"],
                is_available=True
            )
            models.append(model_info)

        return models

    except Exception as e:
        # If we can't get models, return empty list with error info
        return []


@router.get("/", response_model=ModelsResponse)
async def get_available_models(
    backend_type: Optional[str] = Query(
        None,
        description="Filter by backend type (ollama, llama_cpp, custom)"
    )
):
    """
    Get list of available VLM models.

    Returns information about available models including their capabilities
    and which backend they're available on.
    """
    try:
        # Filter by backend if specified
        if backend_type:
            try:
                backend_enum = BackendType(backend_type)
                models = await get_models_for_backend(backend_enum)

                return ModelsResponse(
                    backend_type=backend_enum,
                    models=models,
                    count=len(models),
                    default_model=models[0].name if models else None
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend type: {backend_type}. Available: {[b.value for b in BackendType]}"
                )

        # Get models from all backends
        all_models = []
        backend_types = []

        for backend in BackendType:
            try:
                models = await get_models_for_backend(backend)
                all_models.extend(models)
                if models:
                    backend_types.append(backend)
            except Exception:
                # Skip backends that aren't available
                continue

        return ModelsResponse(
            backend_type=BackendType.OLLAMA,  # Default backend for listing
            models=all_models,
            count=len(all_models),
            default_model=all_models[0].name if all_models else None
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models: {str(e)}"
        )


@router.get("/backends")
async def get_available_backends():
    """
    Get list of available backend types.
    """
    backends = []
    for backend_type in BackendType:
        backend_info = {
            "name": backend_type.value,
            "display_name": backend_type.value.replace("_", " ").title(),
            "description": _get_backend_description(backend_type)
        }
        backends.append(backend_info)

    return {
        "backends": backends,
        "count": len(backends)
    }


@router.get("/backends/{backend_type}")
async def get_backend_models(backend_type: str):
    """
    Get models available for a specific backend.
    """
    try:
        backend_enum = BackendType(backend_type)
        models = await get_models_for_backend(backend_enum)

        return {
            "backend_type": backend_type,
            "backend_name": backend_type.replace("_", " ").title(),
            "models": [model.dict() for model in models],
            "count": len(models),
            "checked_at": datetime.now().isoformat()
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend type: {backend_type}. Available: {[b.value for b in BackendType]}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving models for {backend_type}: {str(e)}"
        )


@router.get("/info/{model_name}")
async def get_model_info(
    model_name: str,
    backend_type: Optional[str] = Query(None, description="Backend type to check")
):
    """
    Get detailed information about a specific model.
    """
    try:
        models_to_check = []

        if backend_type:
            # Check specific backend
            try:
                backend_enum = BackendType(backend_type)
                models_to_check = [(backend_enum, await get_models_for_backend(backend_enum))]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend type: {backend_type}"
                )
        else:
            # Check all backends
            for backend in BackendType:
                try:
                    backend_models = await get_models_for_backend(backend)
                    models_to_check.append((backend, backend_models))
                except Exception:
                    continue

        # Find the model
        found_models = []
        for backend, models in models_to_check:
            for model in models:
                if model.name.lower() == model_name.lower():
                    found_models.append((backend, model))

        if not found_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )

        # Return info for all found instances (model might be available on multiple backends)
        result = {
            "model_name": model_name,
            "instances": [
                {
                    "backend": backend.value,
                    "backend_display_name": backend.value.replace("_", " ").title(),
                    **model.dict()
                }
                for backend, model in found_models
            ],
            "found_in": len(found_models),
            "checked_at": datetime.now().isoformat()
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )


@router.post("/test")
async def test_model_connection(
    model_name: str,
    backend_type: str = BackendType.OLLAMA.value
):
    """
    Test connection to a specific model.
    """
    try:
        backend_enum = BackendType(backend_type)

        # Create converter with specific model
        config = VisionPDFConfig()
        config.vlm.model_name = model_name

        converter = VisionPDF(
            config=config,
            backend_type=backend_enum
        )

        # Test backend connection
        is_connected = await converter.test_backend_connection()

        # Get available models to verify this model exists
        available_models = await converter.get_available_models()
        model_exists = model_name in available_models

        await converter.close()

        return {
            "model_name": model_name,
            "backend_type": backend_type,
            "connection_successful": is_connected,
            "model_available": model_exists,
            "tested_at": datetime.now().isoformat(),
            "status": "available" if is_connected and model_exists else "unavailable"
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend type: {backend_type}"
        )
    except Exception as e:
        return {
            "model_name": model_name,
            "backend_type": backend_type,
            "connection_successful": False,
            "model_available": False,
            "error": str(e),
            "tested_at": datetime.now().isoformat(),
            "status": "error"
        }


def _get_backend_description(backend_type: BackendType) -> str:
    """Get description for a backend type."""
    descriptions = {
        BackendType.OLLAMA: "Local Ollama server with support for various open-source models",
        BackendType.LLAMA_CPP: "llama.cpp backend for efficient local model inference",
        BackendType.CUSTOM: "Custom API backend compatible with OpenAI/Anthropic APIs"
    }
    return descriptions.get(backend_type, "Unknown backend type")
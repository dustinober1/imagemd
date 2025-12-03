"""
Health check and system information routes.

This module provides endpoints for monitoring the API health and status.
"""

import time
import psutil
from datetime import datetime
from typing import Dict, List, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...core.processor import VisionPDF
from ...config.settings import VisionPDFConfig
from ...backends.base import BackendType
from ..models import HealthResponse, ModelInfo

router = APIRouter(prefix="/health", tags=["health"])


async def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_total": memory.total,
            "memory_used": memory.used,
            "disk_percent": disk_percent,
            "disk_total": disk.total,
            "disk_used": disk.used
        }
    except Exception:
        return {}


async def get_backend_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all available backends."""
    backend_status = {}

    for backend_type in BackendType:
        try:
            config = VisionPDFConfig()
            converter = VisionPDF(
                config=config,
                backend_type=backend_type
            )

            # Test backend connection
            is_connected = await converter.test_backend_connection()

            # Get available models
            try:
                models = await converter.get_available_models()
                model_count = len(models)
            except Exception:
                models = []
                model_count = 0

            backend_status[backend_type.value] = {
                "is_connected": is_connected,
                "model_count": model_count,
                "models": models[:5],  # Limit to first 5 models
                "last_checked": datetime.now().isoformat()
            }

            await converter.close()

        except Exception as e:
            backend_status[backend_type.value] = {
                "is_connected": False,
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }

    return backend_status


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns detailed information about:
    - API service status
    - Backend connectivity
    - System metrics
    - Available models
    """
    try:
        # Get system metrics
        system_metrics = await get_system_metrics()

        # Get backend status
        backend_status = await get_backend_status()

        # Get all available models from all backends
        all_models = []
        for backend_info in backend_status.values():
            if backend_info.get("models"):
                all_models.extend(backend_info["models"])

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for model in all_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        # Determine overall health status
        healthy_backends = [
            name for name, info in backend_status.items()
            if info.get("is_connected", False)
        ]

        if healthy_backends:
            status = "healthy"
            message = f"VisionPDF API is healthy with {len(healthy_backends)} active backends"
        else:
            status = "degraded"
            message = "VisionPDF API is running but no backends are available"

        return HealthResponse(
            status=status,
            message=message,
            version="1.0.0",
            uptime=time.time(),  # This would be better tracked at app startup
            system_info={
                **system_metrics,
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                "platform": psutil.platform.platform(),
                "processor_count": psutil.cpu_count(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            available_backends=list(BackendType.__members__.keys()),
            backend_status=backend_status,
            active_jobs=0,  # This would be tracked in a real implementation
            completed_jobs=0,
            failed_jobs=0,
            total_conversions=0
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            version="1.0.0",
            uptime=time.time(),
            system_info={},
            available_backends=[],
            backend_status={},
            active_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            total_conversions=0
        )


@router.get("/simple")
async def simple_health_check():
    """
    Simple health check endpoint.

    Returns a minimal health status suitable for load balancers and monitoring.
    """
    try:
        # Quick backend test
        config = VisionPDFConfig()
        converter = VisionPDF(config=config)
        is_healthy = await converter.test_backend_connection()
        await converter.close()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }

    except Exception:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }


@router.get("/backends")
async def get_backend_status():
    """
    Get detailed status of all VLM backends.
    """
    backend_status = await get_backend_status()
    return {
        "backends": backend_status,
        "checked_at": datetime.now().isoformat()
    }


@router.get("/metrics")
async def get_metrics():
    """
    Get system and performance metrics.
    """
    try:
        system_metrics = await get_system_metrics()
        backend_status = await get_backend_status()

        return {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "backends": {
                name: {
                    "connected": info.get("is_connected", False),
                    "models": info.get("model_count", 0)
                }
                for name, info in backend_status.items()
            }
        }

    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "system": {},
            "backends": {}
        }
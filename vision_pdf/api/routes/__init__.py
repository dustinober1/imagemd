"""
API routes for VisionPDF REST service.

This package contains the route definitions for the FastAPI application.
"""

from .convert import router as convert_router
from .health import router as health_router
from .models import router as models_router
from .jobs import router as jobs_router

__all__ = ["convert_router", "health_router", "models_router", "jobs_router"]
"""
FastAPI REST API for VisionPDF PDF to Markdown conversion service.

This module provides a web API for converting PDF documents to markdown
using the VisionPDF core functionality.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes import convert_router, health_router, models_router, jobs_router

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VisionPDF API",
    description="Convert PDF documents to well-formatted markdown using vision language models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(convert_router)
app.include_router(models_router)
app.include_router(jobs_router)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VisionPDF API",
        "version": "1.0.0",
        "description": "Convert PDF documents to well-formatted markdown using vision language models",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "/health",
            "convert": "/convert",
            "models": "/models",
            "jobs": "/jobs"
        }
    }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "vision_pdf.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
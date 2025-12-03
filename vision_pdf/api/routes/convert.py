"""
PDF conversion routes.

This module provides endpoints for converting PDF files to markdown.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse

from ...core.processor import VisionPDF
from ...config.settings import VisionPDFConfig
from ...backends.base import BackendType, ProcessingMode
from ..models import (
    ConversionConfig, ConversionRequest, ConversionProgress, ConversionResult,
    JobStatus, JobInfo
)
from .jobs import create_job, update_job

router = APIRouter(prefix="/convert", tags=["conversion"])


async def process_pdf_conversion(
    job_id: str,
    pdf_content: bytes,
    filename: str,
    config: ConversionConfig
):
    """
    Background task to process PDF conversion.
    """
    temp_path = None
    converter = None

    try:
        # Update job status
        update_job(job_id, JobStatus.PROCESSING, message="Starting PDF conversion...")

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name

        # Create VisionPDF instance with specified configuration
        vision_config = VisionPDFConfig()

        # Apply configuration
        vision_config.vlm.model_name = config.model_name
        vision_config.processing.preserve_tables = config.preserve_tables
        vision_config.processing.preserve_math = config.preserve_math
        vision_config.processing.preserve_code = config.preserve_code
        vision_config.processing.preserve_images = config.preserve_images
        vision_config.processing.parallel_processing = config.parallel_processing
        vision_config.processing.ocr_fallback = config.ocr_fallback
        vision_config.pdf.rendering_dpi = config.dpi

        if config.max_pages:
            vision_config.processing.max_pages = config.max_pages

        # Create converter
        converter = VisionPDF(
            config=vision_config,
            backend_type=config.backend_type,
            processing_mode=config.processing_mode
        )

        # Create progress callback
        async def progress_callback(progress: float, message: str = "", current_page: Optional[int] = None, total_pages: Optional[int] = None):
            update_job(
                job_id,
                progress=progress,
                message=message
            )

        # Update job with file info
        job = update_job(job_id, message=f"Processing {filename} ({len(pdf_content)} bytes)")

        # Convert PDF
        start_time = asyncio.get_event_loop().time()
        markdown_content = await converter.convert_pdf(
            temp_path,
            progress_callback=progress_callback
        )
        processing_time = asyncio.get_event_loop().time() - start_time

        # Create result
        result = ConversionResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            markdown_content=markdown_content,
            pages_processed=markdown_content.count('\n## Page'),  # Simple page count estimate
            processing_time=processing_time,
            original_filename=filename,
            original_size=len(pdf_content),
            markdown_size=len(markdown_content.encode('utf-8'))
        )

        # Update job with result
        update_job(
            job_id,
            JobStatus.COMPLETED,
            progress=1.0,
            message="Conversion completed successfully",
            result=result
        )

    except Exception as e:
        import traceback
        error_msg = f"Conversion failed: {str(e)}\n{traceback.format_exc()}"

        # Update job with error
        update_job(
            job_id,
            JobStatus.FAILED,
            message=f"Conversion failed: {str(e)}",
            error=error_msg
        )

    finally:
        # Clean up resources
        if converter:
            try:
                await converter.close()
            except Exception:
                pass

        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


@router.post("/")
async def convert_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    backend_type: BackendType = Form(BackendType.OLLAMA),
    processing_mode: ProcessingMode = Form(ProcessingMode.HYBRID),
    model_name: Optional[str] = Form(None),
    preserve_tables: bool = Form(True),
    preserve_math: bool = Form(True),
    preserve_code: bool = Form(True),
    preserve_images: bool = Form(True),
    parallel_processing: bool = Form(True),
    ocr_fallback: bool = Form(True),
    dpi: int = Form(200),
    max_pages: Optional[int] = Form(None)
):
    """
    Convert a PDF file to markdown.

    This endpoint accepts a PDF file upload and creates a conversion job.
    The conversion is processed in the background, and you can track progress
    using the job management endpoints.
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Check file size (50MB limit)
    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")

    # Create conversion configuration
    config = ConversionConfig(
        backend_type=backend_type,
        processing_mode=processing_mode,
        model_name=model_name,
        preserve_tables=preserve_tables,
        preserve_math=preserve_math,
        preserve_code=preserve_code,
        preserve_images=preserve_images,
        parallel_processing=parallel_processing,
        ocr_fallback=ocr_fallback,
        dpi=dpi,
        max_pages=max_pages
    )

    # Create job
    job = create_job(
        config=config,
        filename=file.filename,
        file_size=len(file_content)
    )

    # Start background processing
    background_tasks.add_task(
        process_pdf_conversion,
        job.job_id,
        file_content,
        file.filename,
        config
    )

    return {
        "job_id": job.job_id,
        "status": "pending",
        "message": "PDF conversion job created successfully",
        "filename": file.filename,
        "file_size": len(file_content),
        "config": config.dict(),
        "created_at": job.created_at.isoformat()
    }


@router.post("/sync")
async def convert_pdf_sync(
    file: UploadFile = File(...),
    backend_type: BackendType = Form(BackendType.OLLAMA),
    processing_mode: ProcessingMode = Form(ProcessingMode.HYBRID),
    model_name: Optional[str] = Form(None),
    preserve_tables: bool = Form(True),
    preserve_math: bool = Form(True),
    preserve_code: bool = Form(True),
    preserve_images: bool = Form(True),
    parallel_processing: bool = Form(True),
    ocr_fallback: bool = Form(True),
    dpi: int = Form(200),
    max_pages: Optional[int] = Form(None)
):
    """
    Convert a PDF file to markdown synchronously.

    This endpoint waits for the conversion to complete and returns the result.
    Recommended for small files only as it has a 60-second timeout.
    """
    # Validate file
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Check file size (10MB limit for sync conversion)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit for synchronous conversion. Use /convert for larger files."
        )

    # Create conversion configuration
    config = ConversionConfig(
        backend_type=backend_type,
        processing_mode=processing_mode,
        model_name=model_name,
        preserve_tables=preserve_tables,
        preserve_math=preserve_math,
        preserve_code=preserve_code,
        preserve_images=preserve_images,
        parallel_processing=parallel_processing,
        ocr_fallback=ocr_fallback,
        dpi=dpi,
        max_pages=max_pages
    )

    temp_path = None
    converter = None

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        # Create VisionPDF instance
        vision_config = VisionPDFConfig()

        # Apply configuration
        if config.model_name:
            vision_config.vlm.model_name = config.model_name
        vision_config.processing.preserve_tables = config.preserve_tables
        vision_config.processing.preserve_math = config.preserve_math
        vision_config.processing.preserve_code = config.preserve_code
        vision_config.processing.preserve_images = config.preserve_images
        vision_config.processing.parallel_processing = config.parallel_processing
        vision_config.processing.ocr_fallback = config.ocr_fallback
        vision_config.pdf.rendering_dpi = config.dpi

        if config.max_pages:
            vision_config.processing.max_pages = config.max_pages

        # Create converter
        converter = VisionPDF(
            config=vision_config,
            backend_type=config.backend_type,
            processing_mode=config.processing_mode
        )

        # Convert PDF with timeout
        try:
            markdown_content = await asyncio.wait_for(
                converter.convert_pdf(temp_path),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="Conversion timeout. File may be too large for synchronous processing. Use /convert for background processing."
            )

        # Create result
        result = ConversionResult(
            job_id="sync",
            status=JobStatus.COMPLETED,
            markdown_content=markdown_content,
            pages_processed=markdown_content.count('\n## Page'),
            processing_time=0,  # Not tracking for sync
            original_filename=file.filename,
            original_size=len(file_content),
            markdown_size=len(markdown_content.encode('utf-8'))
        )

        return {
            "status": "completed",
            "message": "PDF converted successfully",
            "result": result.dict(),
            "filename": file.filename,
            "file_size": len(file_content)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )

    finally:
        # Clean up resources
        if converter:
            try:
                await converter.close()
            except Exception:
                pass

        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


@router.get("/validate")
async def validate_conversion_request(
    filename: str,
    file_size: int,
    backend_type: BackendType = BackendType.OLLAMA,
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
):
    """
    Validate a conversion request without uploading the file.

    Useful for clients to check if a conversion request is valid before
    uploading the actual file.
    """
    errors = []
    warnings = []

    # Validate filename
    if not filename.lower().endswith('.pdf'):
        errors.append("File must be a PDF")

    # Validate file size
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        errors.append(f"File size {file_size} bytes exceeds maximum {max_size} bytes")

    # Validate backend availability
    try:
        config = VisionPDFConfig()
        converter = VisionPDF(
            config=config,
            backend_type=backend_type,
            processing_mode=processing_mode
        )
        is_available = await converter.test_backend_connection()
        await converter.close()

        if not is_available:
            errors.append(f"Backend {backend_type.value} is not available")
    except Exception as e:
        warnings.append(f"Could not verify backend availability: {str(e)}")

    # Additional validation for processing mode
    if processing_mode == ProcessingMode.VISION_ONLY and backend_type == BackendType.CUSTOM:
        warnings.append("Custom API backend may not support vision-only processing")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "file_info": {
            "filename": filename,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        },
        "backend_info": {
            "backend_type": backend_type.value,
            "processing_mode": processing_mode.value
        }
    }


@router.get("/formats")
async def get_supported_formats():
    """
    Get information about supported input formats and output options.
    """
    return {
        "input_formats": {
            "pdf": {
                "mime_types": ["application/pdf"],
                "extensions": [".pdf"],
                "max_file_size": "50MB",
                "description": "Portable Document Format files"
            }
        },
        "output_formats": {
            "markdown": {
                "mime_type": "text/markdown",
                "extension": ".md",
                "description": "Markdown with format preservation",
                "features": [
                    "Table detection and formatting",
                    "Mathematical formula conversion to LaTeX",
                    "Code block identification and syntax highlighting",
                    "Image references and captions",
                    "Document structure preservation"
                ]
            }
        },
        "processing_modes": {
            mode.value: {
                "name": mode.value.replace("_", " ").title(),
                "description": _get_processing_mode_description(mode)
            }
            for mode in ProcessingMode
        },
        "backend_types": {
            backend.value: {
                "name": backend.value.replace("_", " ").title(),
                "description": _get_backend_description(backend)
            }
            for backend in BackendType
        }
    }


def _get_processing_mode_description(mode: ProcessingMode) -> str:
    """Get description for processing mode."""
    descriptions = {
        ProcessingMode.VISION_ONLY: "Use only vision model for PDF analysis and conversion",
        ProcessingMode.HYBRID: "Combine vision analysis with text extraction for best results",
        ProcessingMode.TEXT_ONLY: "Use only traditional text extraction methods"
    }
    return descriptions.get(mode, "Unknown processing mode")


def _get_backend_description(backend: BackendType) -> str:
    """Get description for backend type."""
    descriptions = {
        BackendType.OLLAMA: "Local Ollama server with support for various open-source models",
        BackendType.LLAMA_CPP: "llama.cpp backend for efficient local model inference",
        BackendType.CUSTOM: "Custom API backend compatible with OpenAI/Anthropic APIs"
    }
    return descriptions.get(backend, "Unknown backend type")
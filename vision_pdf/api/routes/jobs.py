"""
Job management routes.

This module provides endpoints for managing conversion jobs.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...core.processor import VisionPDF
from ...config.settings import VisionPDFConfig
from ...backends.base import BackendType, ProcessingMode
from ..models import JobInfo, JobStatus, ConversionResult, ConversionConfig

router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory storage for jobs (in production, use a database)
jobs: Dict[str, JobInfo] = {}


@router.get("/", response_model=Dict[str, Any])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """
    List conversion jobs with optional filtering and pagination.
    """
    filtered_jobs = list(jobs.values())

    # Filter by status if specified
    if status:
        try:
            status_enum = JobStatus(status)
            filtered_jobs = [job for job in filtered_jobs if job.status == status_enum]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Available: {[s.value for s in JobStatus]}"
            )

    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x.created_at, reverse=True)

    # Pagination
    total_count = len(filtered_jobs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_jobs = filtered_jobs[start_idx:end_idx]

    return {
        "jobs": [job.dict() for job in page_jobs],
        "count": len(page_jobs),
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "has_next": end_idx < total_count,
        "has_prev": page > 1
    }


@router.get("/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """
    Get detailed information about a specific job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and clean up its resources.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Don't allow deletion of running jobs
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete job that is currently processing"
        )

    # Remove job
    del jobs[job_id]

    return {
        "message": f"Job {job_id} deleted successfully",
        "job_id": job_id,
        "deleted_at": datetime.now().isoformat()
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running or pending job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )

    # Update job status
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now()
    job.message = "Job cancelled by user request"

    return {
        "message": f"Job {job_id} cancelled successfully",
        "job_id": job_id,
        "cancelled_at": datetime.now().isoformat()
    }


@router.get("/{job_id}/result", response_model=ConversionResult)
async def get_job_result(job_id: str):
    """
    Get the conversion result for a completed job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status.value}"
        )

    if not job.result:
        raise HTTPException(status_code=404, detail="No result available for this job")

    return job.result


@router.get("/{job_id}/progress")
async def get_job_progress(job_id: str):
    """
    Get real-time progress information for a job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    progress_info = {
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at.isoformat(),
        "current_page": getattr(job, 'current_page', None),
        "total_pages": getattr(job, 'total_pages', None)
    }

    if job.started_at:
        progress_info["started_at"] = job.started_at.isoformat()
    if job.completed_at:
        progress_info["completed_at"] = job.completed_at.isoformat()

    # Calculate estimated time remaining for processing jobs
    if job.status == JobStatus.PROCESSING and job.started_at:
        elapsed_time = (datetime.now() - job.started_at).total_seconds()
        if job.progress > 0:
            estimated_total_time = elapsed_time / job.progress
            estimated_remaining = estimated_total_time - elapsed_time
            progress_info["estimated_time_remaining"] = max(0, estimated_remaining)

    return progress_info


@router.post("/{job_id}/retry")
async def retry_job(job_id: str):
    """
    Retry a failed job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status: {job.status.value}"
        )

    # Reset job status for retry
    old_job_id = job_id
    new_job_id = str(uuid.uuid4())

    # Create new job with same configuration
    new_job = JobInfo(
        job_id=new_job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        created_at=datetime.now(),
        config=job.config,
        original_filename=job.original_filename,
        file_size=job.file_size
    )

    new_job.message = "Job queued for retry"

    # Store new job
    jobs[new_job_id] = new_job

    return {
        "message": f"Job retry created successfully",
        "old_job_id": old_job_id,
        "new_job_id": new_job_id,
        "retry_created_at": datetime.now().isoformat()
    }


@router.get("/stats/summary")
async def get_job_statistics():
    """
    Get summary statistics for all jobs.
    """
    if not jobs:
        return {
            "total_jobs": 0,
            "status_counts": {},
            "average_processing_time": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "success_rate": 0
        }

    # Count jobs by status
    status_counts = {}
    for status in JobStatus:
        status_counts[status.value] = sum(1 for job in jobs.values() if job.status == status)

    # Calculate processing statistics
    completed_jobs = [job for job in jobs.values() if job.status == JobStatus.COMPLETED]
    failed_jobs = [job for job in jobs.values() if job.status == JobStatus.FAILED]

    total_successful = len(completed_jobs)
    total_failed = len(failed_jobs)
    total_finished = total_successful + total_failed

    # Calculate average processing time
    processing_times = []
    for job in completed_jobs:
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
            processing_times.append(processing_time)

    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

    # Calculate success rate
    success_rate = (total_successful / total_finished * 100) if total_finished > 0 else 0

    return {
        "total_jobs": len(jobs),
        "status_counts": status_counts,
        "average_processing_time": round(avg_processing_time, 2),
        "successful_jobs": total_successful,
        "failed_jobs": total_failed,
        "success_rate": round(success_rate, 2),
        "active_jobs": status_counts.get("processing", 0) + status_counts.get("pending", 0)
    }


@router.delete("/cleanup")
async def cleanup_old_jobs(
    max_age_hours: int = Query(24, ge=1, description="Maximum age in hours for jobs to keep")
):
    """
    Clean up old completed and failed jobs.
    """
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

    jobs_to_delete = []
    for job_id, job in jobs.items():
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            if job.completed_at and job.completed_at.timestamp() < cutoff_time:
                jobs_to_delete.append(job_id)

    # Delete old jobs
    for job_id in jobs_to_delete:
        del jobs[job_id]

    return {
        "message": f"Cleaned up {len(jobs_to_delete)} old jobs",
        "deleted_jobs": jobs_to_delete,
        "max_age_hours": max_age_hours,
        "cleanup_timestamp": datetime.now().isoformat(),
        "remaining_jobs": len(jobs)
    }


# Helper function to create job (used by convert endpoint)
def create_job(
    config: ConversionConfig,
    filename: Optional[str] = None,
    file_size: Optional[int] = None
) -> JobInfo:
    """Create a new job and return its info."""
    job_id = str(uuid.uuid4())

    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        created_at=datetime.now(),
        config=config,
        original_filename=filename,
        file_size=file_size
    )

    job.message = "Job created and queued for processing"

    # Store job
    jobs[job_id] = job

    return job


# Helper function to update job (used by background processing)
def update_job(
    job_id: str,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result: Optional[ConversionResult] = None,
    error: Optional[str] = None
) -> bool:
    """Update job information."""
    if job_id not in jobs:
        return False

    job = jobs[job_id]

    if status is not None:
        job.status = status
        if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = datetime.now()
        elif status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.now()

    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))

    if message is not None:
        job.message = message

    if result is not None:
        job.result = result

    if error is not None:
        job.error = error

    return True


def get_job_sync(job_id: str) -> Optional[JobInfo]:
    """Synchronous version of get_job for internal use."""
    return jobs.get(job_id)
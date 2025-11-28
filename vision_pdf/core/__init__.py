"""
Core processing pipeline and data structures for VisionPDF.

This module contains the main processing logic, document representations,
and orchestration components that drive the PDF to markdown conversion.
"""

from .document import Document, Page, ContentElement, BoundingBox, ContentType, ProcessingMode
from .processor import VisionPDF

__all__ = [
    "Document",
    "Page",
    "ContentElement",
    "BoundingBox",
    "ContentType",
    "ProcessingMode",
    "VisionPDF"
]
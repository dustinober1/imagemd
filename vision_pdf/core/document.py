"""
Core data structures for representing PDF documents and processed content.

This module defines the fundamental data classes used throughout the VisionPDF
package for representing documents, pages, and processed content elements.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import pathlib


class ContentType(Enum):
    """Types of content that can be detected in PDF documents."""
    TEXT = "text"
    TABLE = "table"
    MATHEMATICAL = "mathematical"
    CODE = "code"
    IMAGE = "image"
    LAYOUT = "layout"


class ProcessingMode(Enum):
    """Processing modes for PDF to markdown conversion."""
    VISION_ONLY = "vision_only"  # Convert PDF pages to images and use VLM for complete analysis
    HYBRID = "hybrid"           # Extract text via OCR/PyPDF and use VLM for layout/formatting
    TEXT_ONLY = "text_only"     # Traditional text extraction only (no vision processing)


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box on a page."""
    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge
    page: int  # Page number (0-based)

    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("Invalid bounding box coordinates")
        if self.page < 0:
            raise ValueError("Page number must be non-negative")


@dataclass
class ContentElement:
    """Represents a piece of content extracted from a PDF page."""
    content_type: ContentType
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate content element."""
        if not self.text:
            raise ValueError("Content text cannot be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class Page:
    """Represents a single page from a PDF document."""
    page_number: int  # 0-based page number
    width: float      # Page width in points
    height: float     # Page height in points
    dpi: int          # Resolution used for image conversion
    image_path: Optional[pathlib.Path] = None
    elements: List[ContentElement] = field(default_factory=list)
    raw_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Backward compatibility attribute
    processing_method: Optional[str] = None

    def __post_init__(self):
        """Validate page data."""
        if self.page_number < 0:
            raise ValueError("Page number must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Page dimensions must be positive")
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")

    def get_elements_by_type(self, content_type: ContentType) -> List[ContentElement]:
        """Get all elements of a specific content type."""
        return [elem for elem in self.elements if elem.content_type == content_type]

    def add_element(self, element: ContentElement) -> None:
        """Add a content element to the page."""
        self.elements.append(element)

    def get_text_elements(self) -> List[ContentElement]:
        """Get all text elements from the page."""
        return self.get_elements_by_type(ContentType.TEXT)

    def get_table_elements(self) -> List[ContentElement]:
        """Get all table elements from the page."""
        return self.get_elements_by_type(ContentType.TABLE)

    def get_mathematical_elements(self) -> List[ContentElement]:
        """Get all mathematical expression elements from the page."""
        return self.get_elements_by_type(ContentType.MATHEMATICAL)

    def get_code_elements(self) -> List[ContentElement]:
        """Get all code block elements from the page."""
        return self.get_elements_by_type(ContentType.CODE)


@dataclass
class Document:
    """Represents a complete PDF document."""
    file_path: pathlib.Path
    title: Optional[str] = None
    author: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    pdf_version: Optional[str] = None
    page_count: int = 0
    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate document data."""
        if not self.file_path:
            raise ValueError("File path cannot be empty")
        if self.page_count < 0:
            raise ValueError("Page count must be non-negative")
        # Auto-correct page count if it doesn't match
        if self.page_count != len(self.pages):
            self.page_count = len(self.pages)

    def add_page(self, page: Page) -> None:
        """Add a page to the document."""
        self.pages.append(page)
        self.page_count = len(self.pages)

    def get_page(self, page_number: int) -> Optional[Page]:
        """Get a specific page by number."""
        if 0 <= page_number < len(self.pages):
            return self.pages[page_number]
        return None

    def get_all_elements_by_type(self, content_type: ContentType) -> List[ContentElement]:
        """Get all elements of a specific type across all pages."""
        elements = []
        for page in self.pages:
            elements.extend(page.get_elements_by_type(content_type))
        return elements

    def get_total_elements(self) -> int:
        """Get total number of content elements across all pages."""
        return sum(len(page.elements) for page in self.pages)

    def get_processing_complexity(self) -> str:
        """Estimate processing complexity based on document characteristics."""
        if self.page_count > 100:
            return "high"
        elif self.page_count > 20:
            return "medium"
        else:
            return "low"

    def has_complex_layouts(self) -> bool:
        """Check if document likely contains complex layouts."""
        # Check for multiple pages or elements suggesting complexity
        return (
            self.page_count > 5 or
            any(len(page.elements) > 10 for page in self.pages) or
            any(page.get_table_elements() for page in self.pages) or
            any(page.get_mathematical_elements() for page in self.pages)
        )
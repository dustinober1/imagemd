"""
Pytest configuration and fixtures for VisionPDF tests.

This module provides shared test fixtures and configuration for the
VisionPDF test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import fitz  # PyMuPDF
from PIL import Image
import io

from vision_pdf.config.settings import VisionPDFConfig, ProcessingMode, BackendType
from vision_pdf.core.document import Document, Page, ContentElement, ContentType
from vision_pdf.pdf.renderer import PDFRenderer
from vision_pdf.pdf.analyzer import PDFAnalyzer
from vision_pdf.pdf.extractor import PDFExtractor


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample VisionPDF configuration for testing."""
    config_data = {
        "processing": {
            "mode": ProcessingMode.HYBRID,
            "parallel_processing": True,
            "max_workers": 2,
            "batch_size": 2,
            "dpi": 150,
            "preserve_tables": True,
            "preserve_math": True,
            "preserve_code": True,
            "preserve_images": False
        },
        "default_backend": BackendType.OLLAMA,
        "ocr": {
            "engine": "easyocr",
            "languages": ["en"],
            "confidence_threshold": 0.5,
            "enabled": False  # Disable OCR for faster tests
        },
        "cache": {
            "type": "disabled"
        },
        "logging": {
            "level": "DEBUG"
        },
        "temp_directory": "/tmp/vision_pdf_test"
    }
    return VisionPDFConfig(**config_data)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "sample.pdf"

    # Create a simple PDF with PyMuPDF
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard letter size

    # Add some text content
    page.insert_text((72, 72), "Sample Document Title", fontsize=18, fontname="helv-bold")
    page.insert_text((72, 100), "This is a sample document for testing.", fontsize=12)
    page.insert_text((72, 120), "It contains multiple lines of text.", fontsize=12)

    # Add a list
    page.insert_text((72, 150), "• First item", fontsize=12)
    page.insert_text((72, 170), "• Second item", fontsize=12)
    page.insert_text((72, 190), "• Third item", fontsize=12)

    # Add a simple table-like structure
    page.insert_text((72, 220), "Column 1    Column 2    Column 3", fontsize=12)
    page.insert_text((72, 240), "Value 1     Value 2     Value 3", fontsize=12)
    page.insert_text((72, 260), "Data 1      Data 2      Data 3", fontsize=12)

    # Add some mathematical content
    page.insert_text((72, 300), "Mathematical formula: E = mc²", fontsize=12)

    # Save the PDF
    doc.save(pdf_path)
    doc.close()

    return pdf_path


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image file for testing."""
    image_path = temp_dir / "sample.png"

    # Create a simple test image
    img = Image.new('RGB', (800, 600), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw some test content
    draw.rectangle([50, 50, 750, 550], outline='black', width=2)
    draw.text((100, 100), "Test Image Content", fill='black')
    draw.text((100, 150), "Sample text for OCR testing", fill='black')

    img.save(image_path)
    return image_path


@pytest.fixture
def mock_backend_config():
    """Create a mock backend configuration."""
    return {
        "backend_type": BackendType.OLLAMA,
        "config": {
            "base_url": "http://localhost:11434",
            "model": "test-model",
            "timeout": 30
        },
        "timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 1.0
    }


@pytest.fixture
def mock_vlm_backend():
    """Create a mock VLM backend for testing."""
    from vision_pdf.backends.base import VLMBackend, ProcessingRequest, ProcessingResponse

    class MockVLMBackend(VLMBackend):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False

        async def initialize(self):
            self.initialized = True

        async def cleanup(self):
            self.initialized = False

        async def get_available_models(self):
            return []

        async def test_connection(self):
            return True

        async def process_page(self, request, progress_callback=None):
            # Return a mock response
            return ProcessingResponse(
                markdown="# Mock Page Content\n\nThis is mock markdown content.",
                confidence=0.9,
                processing_time=1.0,
                model_used="mock-model",
                elements_detected=["text", "heading"]
            )

        def _get_required_config_keys(self):
            return ["backend_type"]

    return MockVLMBackend(mock_backend_config())


@pytest.fixture
def sample_document():
    """Create a sample Document object for testing."""
    doc = Document(
        file_path=Path("test.pdf"),
        title="Test Document",
        author="Test Author",
        page_count=2
    )

    # Add first page
    page1 = Page(
        page_number=0,
        width=612,
        height=792,
        dpi=150,
        raw_text="This is page 1 content."
    )
    page1.add_element(ContentElement(
        content_type=ContentType.TEXT,
        text="Page 1 Title",
        confidence=0.95,
        metadata={"font_size": 18}
    ))
    doc.add_page(page1)

    # Add second page
    page2 = Page(
        page_number=1,
        width=612,
        height=792,
        dpi=150,
        raw_text="This is page 2 content."
    )
    page2.add_element(ContentElement(
        content_type=ContentType.TEXT,
        text="Page 2 Content",
        confidence=0.92,
        metadata={"font_size": 12}
    ))
    doc.add_page(page2)

    return doc


@pytest.fixture
def pdf_renderer(sample_config):
    """Create a PDFRenderer instance for testing."""
    return PDFRenderer(sample_config)


@pytest.fixture
def pdf_analyzer(sample_config):
    """Create a PDFAnalyzer instance for testing."""
    return PDFAnalyzer(sample_config)


@pytest.fixture
def pdf_extractor(sample_config):
    """Create a PDFExtractor instance for testing."""
    return PDFExtractor(sample_config)


@pytest.fixture
def mock_ocr_result():
    """Create a mock OCR result for testing."""
    return {
        "text": "Sample OCR text content",
        "confidence": 0.85,
        "words": [
            {"text": "Sample", "confidence": 0.9, "bbox": [0, 0, 50, 20]},
            {"text": "OCR", "confidence": 0.8, "bbox": [55, 0, 90, 20]},
            {"text": "text", "confidence": 0.85, "bbox": [95, 0, 130, 20]},
            {"text": "content", "confidence": 0.82, "bbox": [135, 0, 200, 20]}
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary files after each test."""
    yield
    # Cleanup code here if needed


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama"
    )
    config.addinivalue_line(
        "markers", "requires_llama_cpp: marks tests that require llama.cpp"
    )


# Helper functions for tests
def create_test_page(page_number: int = 0, width: int = 612, height: int = 792) -> Page:
    """Create a test Page object."""
    page = Page(
        page_number=page_number,
        width=width,
        height=height,
        dpi=150
    )
    page.add_element(ContentElement(
        content_type=ContentType.TEXT,
        text=f"Test content for page {page_number}",
        confidence=0.9
    ))
    return page


def create_test_content_element(
    content_type: ContentType = ContentType.TEXT,
    text: str = "Test content",
    confidence: float = 0.9
) -> ContentElement:
    """Create a test ContentElement object."""
    return ContentElement(
        content_type=content_type,
        text=text,
        confidence=confidence
    )


def assert_document_valid(document: Document):
    """Assert that a Document object is valid."""
    assert document.file_path
    assert document.page_count >= 0
    assert len(document.pages) == document.page_count
    assert all(isinstance(page, Page) for page in document.pages)


def assert_page_valid(page: Page):
    """Assert that a Page object is valid."""
    assert page.page_number >= 0
    assert page.width > 0
    assert page.height > 0
    assert page.dpi > 0
    assert all(isinstance(element, ContentElement) for element in page.elements)


def assert_content_element_valid(element: ContentElement):
    """Assert that a ContentElement object is valid."""
    assert element.content_type
    assert element.text
    assert 0 <= element.confidence <= 1
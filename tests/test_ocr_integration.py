"""
Tests for OCR integration functionality.

This module tests the OCR fallback system including
engine initialization, text extraction, and integration
with the main processing pipeline.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from vision_pdf.ocr.base import (
    OCRResult, OCRConfig, OCRFallbackManager, OCRPostProcessor,
    OCREngine
)
from vision_pdf.ocr.engines.tesseract_engine import TesseractEngine
from vision_pdf.core.document import Page, ContentElement, ContentType, BoundingBox
from vision_pdf.core.processor import VisionPDF
from vision_pdf.config.settings import VisionPDFConfig, BackendType, ProcessingMode


class TestOCRConfig:
    """Test OCR configuration."""

    def test_default_config(self):
        """Test default OCR configuration."""
        config = OCRConfig()
        assert config.engine == "tesseract"
        assert config.languages == ["eng"]
        assert config.confidence_threshold == 0.6
        assert config.preprocessing is True
        assert config.deskew is True
        assert config.enhancement is True

    def test_custom_config(self):
        """Test custom OCR configuration."""
        config = OCRConfig(
            engine="easyocr",
            languages=["eng", "spa", "fra"],
            confidence_threshold=0.8,
            preprocessing=False
        )
        assert config.engine == "easyocr"
        assert config.languages == ["eng", "spa", "fra"]
        assert config.confidence_threshold == 0.8
        assert config.preprocessing is False

    def test_languages_post_init(self):
        """Test that languages list is properly initialized."""
        config = OCRConfig()
        assert isinstance(config.languages, list)
        assert len(config.languages) > 0


class TestOCRResult:
    """Test OCR result data structure."""

    def test_ocr_result_creation(self):
        """Test creation of OCR result."""
        result = OCRResult(
            text="Sample text",
            confidence=0.85,
            language="eng"
        )
        assert result.text == "Sample text"
        assert result.confidence == 0.85
        assert result.language == "eng"
        assert result.bounding_boxes == []
        assert result.metadata == {}

    def test_ocr_result_with_metadata(self):
        """Test OCR result with metadata."""
        bbox = [(0, 0, 100, 50)]
        metadata = {"engine": "tesseract", "word_count": 2}
        result = OCRResult(
            text="Sample text",
            confidence=0.85,
            language="eng",
            bounding_boxes=bbox,
            metadata=metadata
        )
        assert result.bounding_boxes == bbox
        assert result.metadata == metadata


class MockOCREngine(OCREngine):
    """Mock OCR engine for testing."""

    def _initialize_engine(self):
        """Mock initialization."""
        self.available = True

    def is_available(self):
        """Mock availability check."""
        return self.available

    def extract_text_from_image(self, image_path):
        """Mock text extraction."""
        return OCRResult(
            text="Mock extracted text",
            confidence=0.9,
            language="eng"
        )

    def extract_text_with_layout(self, image_path):
        """Mock layout-aware text extraction."""
        element = ContentElement(
            text="Mock extracted text",
            content_type=ContentType.TEXT,
            confidence=0.9,
            bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=50)
        )
        return [element]


class TestOCRFallbackManager:
    """Test OCR fallback manager."""

    def test_manager_initialization(self):
        """Test OCR fallback manager initialization."""
        config = OCRConfig()
        manager = OCRFallbackManager(config)
        assert manager.config == config
        assert isinstance(manager.engines, dict)

    @patch('vision_pdf.ocr.engines.tesseract_engine.TesseractEngine')
    def test_engine_initialization(self, mock_tesseract_class):
        """Test that engines are properly initialized."""
        # Mock Tesseract engine
        mock_engine = Mock()
        mock_engine.is_available.return_value = True
        mock_tesseract_class.return_value = mock_engine

        config = OCRConfig()
        manager = OCRFallbackManager(config)

        assert "tesseract" in manager.engines
        mock_tesseract_class.assert_called_once_with(config)

    def test_get_available_engines(self):
        """Test getting list of available engines."""
        config = OCRConfig()
        manager = OCRFallbackManager(config)

        # Add mock engine
        mock_engine = MockOCREngine(config)
        manager.engines["mock"] = mock_engine

        available = manager.get_available_engines()
        assert "mock" in available

    def test_process_page_with_ocr(self):
        """Test processing a page with OCR."""
        config = OCRConfig()
        manager = OCRFallbackManager(config)

        # Add mock engine
        mock_engine = MockOCREngine(config)
        manager.engines["mock"] = mock_engine

        # Create test page
        page = Page(
            page_number=0,
            image_path=str(Path("/tmp/test.png"))
        )
        page.elements = []  # Start with no elements

        # Process page
        result_page = manager.process_page_with_ocr(page, "mock")

        assert len(result_page.elements) > 0
        assert result_page.processing_method == "ocr_fallback"
        assert result_page.metadata["ocr_engine"] == "mock"

    def test_should_use_ocr_fallback(self):
        """Test OCR fallback decision logic."""
        config = OCRConfig()
        manager = OCRFallbackManager(config)

        # Test failed processing page
        failed_page = Page(page_number=0)
        failed_page.processing_method = "failed"
        assert manager.should_use_ocr_fallback(failed_page) is True

        # Test already processed page
        ocr_page = Page(page_number=0)
        ocr_page.processing_method = "ocr_fallback"
        assert manager.should_use_ocr_fallback(ocr_page) is False

        # Test low confidence page
        low_conf_page = Page(page_number=0)
        low_conf_page.elements = [
            ContentElement(text="Low confidence", content_type=ContentType.TEXT, confidence=0.3)
        ]
        assert manager.should_use_ocr_fallback(low_conf_page) is True

        # Test high confidence page
        high_conf_page = Page(page_number=0)
        high_conf_page.elements = [
            ContentElement(text="High confidence", content_type=ContentType.TEXT, confidence=0.9)
        ]
        assert manager.should_use_ocr_fallback(high_conf_page) is False

    def test_extract_text_only(self):
        """Test extracting text from image."""
        config = OCRConfig()
        manager = OCRFallbackManager(config)

        # Add mock engine
        mock_engine = MockOCREngine(config)
        manager.engines["mock"] = mock_engine

        result = manager.extract_text_only("/tmp/test.png", "mock")
        assert isinstance(result, OCRResult)
        assert result.text == "Mock extracted text"
        assert result.confidence == 0.9


class TestOCRPostProcessor:
    """Test OCR post-processor."""

    def test_process_page_elements(self):
        """Test processing OCR-extracted elements."""
        config = OCRConfig()
        processor = OCRPostProcessor(config)

        # Create test elements
        elements = [
            ContentElement(
                text="  def hello():  ",
                content_type=ContentType.TEXT,
                confidence=0.7
            ),
            ContentElement(
                text="x^2 + y^2 = z^2",
                content_type=ContentType.TEXT,
                confidence=0.8
            )
        ]

        processed = processor.process_page_elements(elements)

        assert len(processed) == 2
        # Text should be cleaned
        assert processed[0].text == "def hello():"
        assert processed[0].content_type == ContentType.CODE
        assert processed[1].content_type == ContentType.MATHEMATICAL
        assert processed[0].metadata["ocr_processed"] is True

    def test_clean_text(self):
        """Test text cleaning functionality."""
        config = OCRConfig()
        processor = OCRPostProcessor(config)

        # Test excessive whitespace removal
        messy_text = "Multiple     spaces   and\t\ttabs"
        clean = processor._clean_text(messy_text)
        assert clean == "Multiple spaces and tabs"

        # Test leading/trailing whitespace
        messy_text = "   surrounded by spaces   "
        clean = processor._clean_text(messy_text)
        assert clean == "surrounded by spaces"

    def test_detect_content_type(self):
        """Test content type detection."""
        config = OCRConfig()
        processor = OCRPostProcessor(config)

        # Test table detection
        table_text = "Name\tAge\tCity\nJohn\t25\tNYC"
        assert processor._detect_content_type(table_text) == ContentType.TABLE

        # Test code detection
        code_text = "def function(): pass"
        assert processor._detect_content_type(code_text) == ContentType.CODE

        # Test math detection
        math_text = "The formula is x^2 + y^2 = z^2"
        assert processor._detect_content_type(math_text) == ContentType.MATHEMATICAL

        # Test default text
        plain_text = "This is just regular text"
        assert processor._detect_content_type(plain_text) == ContentType.TEXT


class TestTesseractEngine:
    """Test Tesseract OCR engine."""

    @patch('vision_pdf.ocr.engines.tesseract_engine.pytesseract')
    def test_engine_initialization(self, mock_pytesseract):
        """Test Tesseract engine initialization."""
        mock_pytesseract.get_tesseract_version.return_value = "5.0.0"
        config = OCRConfig()

        with patch.dict('os.environ', {'TESSERACT_CMD': '/usr/bin/tesseract'}):
            engine = TesseractEngine(config)
            assert engine.is_available() is True

    @patch('vision_pdf.ocr.engines.tesseract_engine.pytesseract')
    def test_extract_text_from_image(self, mock_pytesseract):
        """Test text extraction from image."""
        # Mock pytesseract results
        mock_data = {
            'text': ['Hello', 'World'],
            'conf': [95, 90],
            'left': [10, 50],
            'top': [10, 10],
            'width': [30, 40],
            'height': [15, 15],
            'line_num': [1, 1]
        }
        mock_pytesseract.image_to_data.return_value = mock_data
        mock_pytesseract.get_tesseract_version.return_value = "5.0.0"

        config = OCRConfig()
        engine = TesseractEngine(config)

        with patch('vision_pdf.ocr.engines.tesseract_engine.Image'):
            result = engine.extract_text_from_image("/tmp/test.png")

        assert "Hello" in result.text
        assert "World" in result.text
        assert result.confidence > 0.8


class TestVisionPdfOCRIntegration:
    """Test VisionPDF OCR integration."""

    @pytest.fixture
    def vision_pdf_with_ocr(self):
        """Create VisionPDF instance with OCR enabled."""
        config = VisionPDFConfig()
        config.processing.ocr_fallback_enabled = True
        config.processing.ocr_config = {
            "engine": "tesseract",
            "languages": ["eng"],
            "confidence_threshold": 0.6
        }

        return VisionPDF(
            config=config,
            backend_type=BackendType.OLLAMA,
            processing_mode=ProcessingMode.HYBRID
        )

    @patch('vision_pdf.ocr.engines.tesseract_engine.TesseractEngine')
    def test_ocr_initialization(self, mock_tesseract_class, vision_pdf_with_ocr):
        """Test that OCR is properly initialized in VisionPDF."""
        # Mock Tesseract engine
        mock_engine = Mock()
        mock_engine.is_available.return_value = True
        mock_tesseract_class.return_value = mock_engine

        # Re-initialize to trigger OCR setup
        vision_pdf_with_ocr._init_ocr_fallback()

        assert vision_pdf_with_ocr.ocr_manager is not None
        assert vision_pdf_with_ocr.ocr_post_processor is not None

    def test_should_use_ocr_fallback(self, vision_pdf_with_ocr):
        """Test OCR fallback decision in VisionPDF."""
        # Create a test page with low confidence
        low_conf_page = Page(page_number=0)
        low_conf_page.elements = [
            ContentElement(text="Low confidence", content_type=ContentType.TEXT, confidence=0.3)
        ]

        # Mock OCR manager
        vision_pdf_with_ocr.ocr_manager = Mock()
        vision_pdf_with_ocr.ocr_manager.should_use_ocr_fallback.return_value = True

        result = vision_pdf_with_ocr._should_use_ocr_fallback(low_conf_page)
        assert result is True
        vision_pdf_with_ocr.ocr_manager.should_use_ocr_fallback.assert_called_once_with(low_conf_page)

    @patch('vision_pdf.ocr.engines.tesseract_engine.TesseractEngine')
    async def test_ocr_fallback_processing(self, mock_tesseract_class, vision_pdf_with_ocr):
        """Test OCR fallback processing."""
        # Mock Tesseract engine and results
        mock_engine = Mock()
        mock_engine.is_available.return_value = True
        mock_tesseract_class.return_value = mock_engine

        # Mock OCR manager
        mock_ocr_manager = Mock()
        mock_processed_page = Page(page_number=0)
        mock_processed_page.elements = [
            ContentElement(text="OCR extracted", content_type=ContentType.TEXT, confidence=0.9)
        ]
        mock_ocr_manager.process_page_with_ocr.return_value = mock_processed_page

        vision_pdf_with_ocr.ocr_manager = mock_ocr_manager
        vision_pdf_with_ocr.ocr_post_processor = Mock()

        # Test page
        test_page = Page(page_number=0)

        # Process with OCR fallback
        result = await vision_pdf_with_ocr._ocr_fallback_processing(test_page)

        assert isinstance(result, str)
        mock_ocr_manager.process_page_with_ocr.assert_called_once_with(test_page)

    def test_ocr_disabled(self):
        """Test behavior when OCR is disabled."""
        config = VisionPDFConfig()
        config.processing.ocr_fallback_enabled = False

        vision_pdf = VisionPDF(
            config=config,
            backend_type=BackendType.OLLAMA
        )

        vision_pdf._init_ocr_fallback()

        assert vision_pdf.ocr_manager is None
        assert vision_pdf.ocr_post_processor is None


if __name__ == "__main__":
    pytest.main([__file__])
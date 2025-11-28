"""
Base classes for OCR integration in VisionPDF.

This module provides the abstract interfaces and common functionality
for OCR processing as a fallback when VLM processing fails.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..core.document import Document, Page, ContentElement, ContentType
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str
    confidence: float
    language: Optional[str] = None
    bounding_boxes: List[Tuple[int, int, int, int]] = None  # List of (x0, y0, x1, y1)
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize OCR result."""
        if self.bounding_boxes is None:
            self.bounding_boxes = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    engine: str = "tesseract"  # tesseract, easyocr, paddleocr
    languages: List[str] = None
    confidence_threshold: float = 0.6
    preprocessing: bool = True
    deskew: bool = True
    enhancement: bool = True

    def __post_init__(self):
        """Initialize OCR config."""
        if self.languages is None:
            self.languages = ["eng"]


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    def __init__(self, config: OCRConfig):
        """
        Initialize OCR engine.

        Args:
            config: OCR configuration
        """
        self.config = config
        self._initialize_engine()

    @abstractmethod
    def _initialize_engine(self) -> None:
        """Initialize the OCR engine with required dependencies."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the OCR engine is available and properly configured.

        Returns:
            True if engine is available
        """
        pass

    @abstractmethod
    def extract_text_from_image(self, image_path: str) -> OCRResult:
        """
        Extract text from an image file.

        Args:
            image_path: Path to the image file

        Returns:
            OCR result with extracted text and metadata
        """
        pass

    @abstractmethod
    def extract_text_with_layout(self, image_path: str) -> List[ContentElement]:
        """
        Extract text with layout information.

        Args:
            image_path: Path to the image file

        Returns:
            List of content elements with layout information
        """
        pass

    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results.

        Args:
            image_path: Path to the original image

        Returns:
            Path to preprocessed image
        """
        # Basic preprocessing - can be overridden by subclasses
        return image_path

    def validate_result(self, result: OCRResult) -> bool:
        """
        Validate OCR result quality.

        Args:
            result: OCR result to validate

        Returns:
            True if result meets quality criteria
        """
        return (
            result.confidence >= self.config.confidence_threshold and
            len(result.text.strip()) > 0
        )


class OCRFallbackManager:
    """Manages OCR fallback processing for failed VLM operations."""

    def __init__(self, config: OCRConfig):
        """
        Initialize OCR fallback manager.

        Args:
            config: OCR configuration
        """
        self.config = config
        self.engines = {}
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize available OCR engines."""
        try:
            from .engines.tesseract_engine import TesseractEngine
            self.engines["tesseract"] = TesseractEngine(self.config)
            logger.info("Tesseract OCR engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Tesseract: {e}")

        try:
            from .engines.easyocr_engine import EasyOCREngine
            self.engines["easyocr"] = EasyOCREngine(self.config)
            logger.info("EasyOCR engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")

        try:
            from .engines.paddleocr_engine import PaddleOCREngine
            self.engines["paddleocr"] = PaddleOCREngine(self.config)
            logger.info("PaddleOCR engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")

    def get_available_engines(self) -> List[str]:
        """
        Get list of available OCR engines.

        Returns:
            List of available engine names
        """
        return [name for name, engine in self.engines.items() if engine.is_available()]

    def process_page_with_ocr(
        self,
        page: Page,
        preferred_engine: Optional[str] = None
    ) -> Page:
        """
        Process a page using OCR as fallback.

        Args:
            page: Page to process
            preferred_engine: Preferred OCR engine to use

        Returns:
            Page with OCR-processed content
        """
        available_engines = self.get_available_engines()
        if not available_engines:
            logger.error("No OCR engines available for fallback processing")
            return page

        # Choose engine
        engine_name = preferred_engine or self.config.engine
        if engine_name not in available_engines:
            engine_name = available_engines[0]
            logger.info(f"Using fallback OCR engine: {engine_name}")

        engine = self.engines[engine_name]

        try:
            # Process page image
            if page.image_path:
                content_elements = engine.extract_text_with_layout(page.image_path)

                # Update page with OCR results
                page.elements.extend(content_elements)
                page.processing_method = "ocr_fallback"
                page.metadata["ocr_engine"] = engine_name
                page.metadata["ocr_confidence"] = (
                    sum(el.confidence for el in content_elements) / len(content_elements)
                    if content_elements else 0.0
                )

                logger.info(f"OCR fallback processed {len(content_elements)} elements")
            else:
                logger.warning("No image available for OCR processing")

        except Exception as e:
            logger.error(f"OCR fallback processing failed: {e}")

        return page

    def extract_text_only(
        self,
        image_path: str,
        preferred_engine: Optional[str] = None
    ) -> OCRResult:
        """
        Extract text from image using OCR.

        Args:
            image_path: Path to image
            preferred_engine: Preferred OCR engine

        Returns:
            OCR result with extracted text
        """
        available_engines = self.get_available_engines()
        if not available_engines:
            raise ValueError("No OCR engines available")

        engine_name = preferred_engine or self.config.engine
        if engine_name not in available_engines:
            engine_name = available_engines[0]

        engine = self.engines[engine_name]
        return engine.extract_text_from_image(image_path)

    def should_use_ocr_fallback(
        self,
        page: Page,
        vlm_confidence_threshold: float = 0.5
    ) -> bool:
        """
        Determine if OCR fallback should be used.

        Args:
            page: Page to evaluate
            vlm_confidence_threshold: Minimum VLM confidence threshold

        Returns:
            True if OCR fallback is recommended
        """
        # Use OCR if VLM processing failed or produced low confidence results
        if page.processing_method == "failed":
            return True

        if page.processing_method == "ocr_fallback":
            return False  # Already processed with OCR

        # Calculate average confidence
        if page.elements:
            avg_confidence = sum(el.confidence for el in page.elements) / len(page.elements)
            if avg_confidence < vlm_confidence_threshold:
                return True

        # Use OCR if no content was extracted
        if not page.elements and not page.raw_text:
            return True

        return False


class OCRPostProcessor:
    """Post-processes OCR results for better quality."""

    def __init__(self, config: OCRConfig):
        """
        Initialize OCR post-processor.

        Args:
            config: OCR configuration
        """
        self.config = config

    def process_page_elements(self, elements: List[ContentElement]) -> List[ContentElement]:
        """
        Process and improve OCR-extracted content elements.

        Args:
            elements: List of content elements from OCR

        Returns:
            Improved list of content elements
        """
        processed_elements = []

        for element in elements:
            # Clean up text
            cleaned_text = self._clean_text(element.text)

            # Detect content type based on text patterns
            content_type = self._detect_content_type(cleaned_text)

            # Create improved element
            improved_element = ContentElement(
                text=cleaned_text,
                content_type=content_type,
                confidence=element.confidence,
                bounding_box=element.bounding_box,
                metadata={
                    **element.metadata,
                    "ocr_processed": True,
                    "original_confidence": element.confidence
                }
            )

            processed_elements.append(improved_element)

        return processed_elements

    def _clean_text(self, text: str) -> str:
        """
        Clean up OCR-extracted text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Fix common OCR errors
        corrections = {
            'rn': 'm',
            'vv': 'w',
            'cl': 'd',
            'O': '0',  # Only in numeric contexts
            'l': '1',  # Only in numeric contexts
        }

        # Apply corrections conservatively
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

        return text

    def _detect_content_type(self, text: str) -> ContentType:
        """
        Detect content type from text patterns.

        Args:
            text: Text to analyze

        Returns:
            Detected content type
        """
        import re

        # Check for table patterns
        if re.search(r'(\t|\s{3,}|,)\s*\w+', text) and text.count('\n') > 1:
            return ContentType.TABLE

        # Check for code patterns
        code_indicators = ['def ', 'function', 'var ', 'let ', 'const ', 'import ', 'if ', 'for ']
        if any(indicator in text for indicator in code_indicators):
            return ContentType.CODE

        # Check for mathematical patterns
        math_patterns = [r'\$[^$]+\$', r'\\[a-zA-Z]+', r'[α-ωΑ-Ω]', r'[∑∫∏∂∇]']
        if any(re.search(pattern, text) for pattern in math_patterns):
            return ContentType.MATHEMATICAL

        # Default to text
        return ContentType.TEXT
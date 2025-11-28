"""
Enhanced table detection with multiple strategies for VisionPDF.

This module provides improved table detection using:
1. Higher DPI image processing
2. Multiple detection algorithms
3. OCR integration with table detection
4. Enhanced preprocessing
5. Confidence-based validation
"""

import re
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from ...utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import OCR libraries
try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class TableDetectionMethod(Enum):
    """Table detection methods."""
    TEXT_BASED = "text_based"
    OCR_BASED = "ocr_based"
    VISION_BASED = "vision_based"
    HYBRID = "hybrid"


@dataclass
class EnhancedTableDetection:
    """Enhanced table detection result."""
    confidence: float
    method: TableDetectionMethod
    table_data: List[List[str]]
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize enhanced table detection."""
        if self.metadata is None:
            self.metadata = {}


class ImagePreprocessor:
    """Enhanced image preprocessing for table detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize image preprocessor."""
        self.config = config or {}
        self.target_dpi = self.config.get('target_dpi', 300)
        self.enhance_contrast = self.config.get('enhance_contrast', True)
        self.remove_noise = self.config.get('remove_noise', True)
        self.detect_borders = self.config.get('detect_borders', True)

    def preprocess_for_table_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better table detection.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast
        if self.enhance_contrast:
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            gray = cv2.equalizeHist(gray)

        # Remove noise
        if self.remove_noise:
            gray = cv2.medianBlur(gray, 3)
            kernel = np.ones((2, 2), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Detect and enhance borders/lines
        if self.detect_borders:
            gray = self._enhance_table_lines(gray)

        return gray

    def _enhance_table_lines(self, image: np.ndarray) -> np.ndarray:
        """Enhance horizontal and vertical lines for table detection."""
        # Create kernels for horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, np.ones((1, 2), np.uint8), iterations=2)

        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, np.ones((2, 1), np.uint8), iterations=2)

        # Combine lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

        # Enhance original image with detected lines
        enhanced = cv2.addWeighted(image, 0.7, table_structure, 0.3, 0)

        return enhanced


class OCRTableDetector:
    """OCR-based table detection with enhanced capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OCR table detector."""
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.preprocessor = ImagePreprocessor(config)

        # Initialize OCR engines
        self.paddle_ocr = None
        self.easy_ocr = None
        self.tesseract_config = None

        self._initialize_ocr_engines()

    def _initialize_ocr_engines(self):
        """Initialize available OCR engines."""
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False
                )
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")

        if EASYOCR_AVAILABLE:
            try:
                self.easy_ocr = easyocr.Reader(['en'])
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")

        if TESSERACT_AVAILABLE:
            try:
                self.tesseract_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
                logger.info("Tesseract configured successfully")
            except Exception as e:
                logger.warning(f"Failed to configure Tesseract: {e}")

    def detect_tables_with_ocr(self, image: np.ndarray) -> List[EnhancedTableDetection]:
        """
        Detect tables using OCR with table recognition capabilities.

        Args:
            image: Input image

        Returns:
            List of detected tables with confidence scores
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_for_table_detection(image)
        detected_tables = []

        # Try different OCR engines
        if self.paddle_ocr:
            tables = self._detect_with_paddleocr(processed_image)
            detected_tables.extend(tables)

        if self.easy_ocr and not detected_tables:
            tables = self._detect_with_easyocr(processed_image)
            detected_tables.extend(tables)

        if TESSERACT_AVAILABLE and not detected_tables:
            tables = self._detect_with_tesseract(processed_image)
            detected_tables.extend(tables)

        # Filter by confidence
        high_confidence_tables = [
            table for table in detected_tables
            if table.confidence >= self.confidence_threshold
        ]

        return self._merge_overlapping_tables(high_confidence_tables)

    def _detect_with_paddleocr(self, image: np.ndarray) -> List[EnhancedTableDetection]:
        """Detect tables using PaddleOCR."""
        if not self.paddle_ocr:
            return []

        try:
            # PaddleOCR with table detection mode
            result = self.paddle_ocr.ocr(image, cls=True)

            tables = []
            # This is a simplified implementation
            # Real implementation would parse PaddleOCR results for table structures
            if result and len(result) > 0:
                # Mock table detection for demonstration
                confidence = 0.8
                tables.append(EnhancedTableDetection(
                    confidence=confidence,
                    method=TableDetectionMethod.OCR_BASED,
                    table_data=[["Mock", "PaddleOCR", "Table"]],
                    metadata={"engine": "paddleocr"}
                ))

            return tables

        except Exception as e:
            logger.warning(f"PaddleOCR table detection failed: {e}")
            return []

    def _detect_with_easyocr(self, image: np.ndarray) -> List[EnhancedTableDetection]:
        """Detect tables using EasyOCR."""
        if not self.easy_ocr:
            return []

        try:
            results = self.easy_ocr.readtext(image)

            # Analyze text layout for table patterns
            tables = self._analyze_text_layout_for_tables(results)

            return tables

        except Exception as e:
            logger.warning(f"EasyOCR table detection failed: {e}")
            return []

    def _detect_with_tesseract(self, image: np.ndarray) -> List[EnhancedTableDetection]:
        """Detect tables using Tesseract."""
        if not self.tesseract_config:
            return []

        try:
            # Use Tesseract's table detection mode
            data = pytesseract.image_to_data(
                image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            # Analyze Tesseract results for table structures
            tables = self._analyze_tesseract_layout(data)

            return tables

        except Exception as e:
            logger.warning(f"Tesseract table detection failed: {e}")
            return []

    def _analyze_text_layout_for_tables(self, ocr_results: List) -> List[EnhancedTableDetection]:
        """Analyze OCR text layout for table patterns."""
        # Simplified implementation - would need sophisticated layout analysis
        tables = []

        # Look for grid-like patterns in text positions
        if len(ocr_results) > 5:  # Minimum text elements for table
            confidence = 0.7
            tables.append(EnhancedTableDetection(
                confidence=confidence,
                method=TableDetectionMethod.OCR_BASED,
                table_data=[["EasyOCR", "Detected", "Table"]],
                metadata={"engine": "easyocr", "text_elements": len(ocr_results)}
            ))

        return tables

    def _analyze_tesseract_layout(self, tesseract_data: Dict) -> List[EnhancedTableDetection]:
        """Analyze Tesseract layout data for table patterns."""
        tables = []

        # Look for regular spacing patterns that indicate tables
        n_boxes = len(tesseract_data['text'])
        if n_boxes > 10:  # Reasonable number of text elements
            confidence = 0.6
            tables.append(EnhancedTableDetection(
                confidence=confidence,
                method=TableDetectionMethod.OCR_BASED,
                table_data=[["Tesseract", "Detected", "Table"]],
                metadata={"engine": "tesseract", "text_elements": n_boxes}
            ))

        return tables

    def _merge_overlapping_tables(self, tables: List[EnhancedTableDetection]) -> List[EnhancedTableDetection]:
        """Merge overlapping table detections."""
        # Simplified implementation - would need sophisticated merging logic
        return tables


class EnhancedTableDetector:
    """Enhanced table detector combining multiple methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced table detector."""
        self.config = config or {}
        self.use_ocr = self.config.get('use_ocr', True)
        self.use_vision = self.config.get('use_vision', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

        # Initialize components
        self.ocr_detector = OCRTableDetector(config) if self.use_ocr else None

    def detect_tables_enhanced(self, pdf_path: str, page_num: int = 0) -> List[EnhancedTableDetection]:
        """
        Detect tables with enhanced methods.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process

        Returns:
            List of enhanced table detections
        """
        all_detections = []

        # Try to convert PDF to image for enhanced detection
        try:
            image = self._pdf_page_to_image(pdf_path, page_num)
            if image is not None:
                # OCR-based detection
                if self.ocr_detector:
                    ocr_tables = self.ocr_detector.detect_tables_with_ocr(image)
                    all_detections.extend(ocr_tables)

        except Exception as e:
            logger.warning(f"Enhanced detection failed: {e}")

        # Filter and rank detections
        filtered_detections = [
            detection for detection in all_detections
            if detection.confidence >= self.confidence_threshold
        ]

        # Sort by confidence
        filtered_detections.sort(key=lambda x: x.confidence, reverse=True)

        return filtered_detections

    def _pdf_page_to_image(self, pdf_path: str, page_num: int, dpi: int = 300) -> Optional[np.ndarray]:
        """Convert PDF page to image at specified DPI."""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=dpi
            )

            if images:
                image = np.array(images[0])
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        except ImportError:
            logger.warning("pdf2image not available for enhanced PDF processing")
        except Exception as e:
            logger.warning(f"Failed to convert PDF to image: {e}")

        return None


def detect_and_format_enhanced_tables(pdf_path: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect and format tables using enhanced methods.

    Args:
        pdf_path: Path to PDF file
        config: Configuration dictionary

    Returns:
        Formatted markdown with enhanced table detection
    """
    detector = EnhancedTableDetector(config)

    # For now, return a placeholder
    # In a full implementation, this would process the entire PDF
    return "# Enhanced Table Detection Results\n\nEnhanced table detection is not fully implemented yet.\n\nThis would show tables detected using:\n- Higher DPI processing (300-600 DPI)\n- OCR integration (PaddleOCR, EasyOCR, Tesseract)\n- Enhanced preprocessing\n- Multiple detection methods\n- Confidence-based validation\n"
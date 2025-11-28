"""
PaddleOCR engine implementation.

This module provides PaddleOCR integration for VisionPDF,
offering high-precision OCR with multilingual support.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    from paddleocr import PaddleOCR
    import cv2
    import numpy as np
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    cv2 = None
    np = None
    PADDLEOCR_AVAILABLE = False

from ..base import OCREngine, OCRResult, ContentElement, ContentType
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation."""

    def __init__(self, config):
        """
        Initialize PaddleOCR engine.

        Args:
            config: OCR configuration
        """
        self.ocr = None
        self.supported_languages = [
            'ch', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'ja', 'ko'
        ]
        super().__init__(config)

    def _initialize_engine(self) -> None:
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR

            # Convert language codes to PaddleOCR format
            lang = self._convert_language(self.config.languages[0] if self.config.languages else 'eng')

            # Initialize PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=False,  # Use CPU for compatibility
                show_log=False
            )

            logger.info(f"PaddleOCR initialized with language: {lang}")

        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def _convert_language(self, language: str) -> str:
        """Convert language code to PaddleOCR format."""
        lang_mapping = {
            'eng': 'en',
            'chi_sim': 'ch',
            'chi_tra': 'chinese_cht',
            'jpn': 'japan',
            'kor': 'korean',
            'spa': 'en',  # PaddleOCR limited multilingual support
            'fra': 'en',
            'deu': 'en',
            'ita': 'en',
            'por': 'en'
        }

        return lang_mapping.get(language, 'en')  # Default to English

    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        try:
            from paddleocr import PaddleOCR
            return self.ocr is not None
        except Exception:
            return False

    def extract_text_from_image(self, image_path: str) -> OCRResult:
        """
        Extract text from image using PaddleOCR.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        if not self.ocr:
            return OCRResult(text="", confidence=0.0, metadata={'error': 'PaddleOCR not initialized'})

        try:
            # Preprocess image if requested
            processed_image_path = self._preprocess_image(image_path)

            # Extract text using PaddleOCR
            results = self.ocr.ocr(processed_image_path, cls=True)

            if not results or not results[0]:
                return OCRResult(text="", confidence=0.0, metadata={'warning': 'No text detected'})

            # Process results
            text_parts = []
            confidences = []
            bounding_boxes = []

            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    if text.strip():
                        text_parts.append(text.strip())
                        confidences.append(confidence)

                        # Convert bounding box format
                        # PaddleOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        flat_bbox = [coord for point in bbox for coord in point]
                        x_coords = flat_bbox[::2]
                        y_coords = flat_bbox[1::2]
                        bounding_boxes.append((
                            min(x_coords), min(y_coords),
                            max(x_coords), max(y_coords)
                        ))

            # Calculate overall confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = ' '.join(text_parts)

            # Clean up temporary files
            if processed_image_path != image_path:
                try:
                    os.unlink(processed_image_path)
                except:
                    pass

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=','.join(self.config.languages),
                bounding_boxes=bounding_boxes,
                metadata={
                    'engine': 'paddleocr',
                    'detection_count': len(text_parts),
                    'languages_used': self.config.languages
                }
            )

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return OCRResult(text="", confidence=0.0, metadata={'error': str(e)})

    def extract_text_with_layout(self, image_path: str) -> List[ContentElement]:
        """
        Extract text with layout information.

        Args:
            image_path: Path to image file

        Returns:
            List of content elements with layout info
        """
        if not self.ocr:
            return []

        try:
            # Preprocess image
            processed_image_path = self._preprocess_image(image_path)

            # Extract text with detailed information
            results = self.ocr.ocr(processed_image_path, cls=True)

            # Convert results to content elements
            elements = self._convert_results_to_elements(results[0] if results else [])

            # Clean up temporary files
            if processed_image_path != image_path:
                try:
                    os.unlink(processed_image_path)
                except:
                    pass

            return elements

        except Exception as e:
            logger.error(f"PaddleOCR layout analysis failed: {e}")
            return []

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results.

        Args:
            image_path: Path to original image

        Returns:
            Path to preprocessed image
        """
        if not self.config.preprocessing or not PADDLEOCR_AVAILABLE:
            return image_path

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return image_path

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)

            # Enhance contrast
            if self.config.enhancement:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            else:
                enhanced = denoised

            # Deskew if requested
            if self.config.deskew:
                enhanced = self._deskew_image(enhanced)

            # Save processed image
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(temp_file.name, enhanced)

            return temp_file.name

        except Exception as e:
            logger.warning(f"PaddleOCR image preprocessing failed: {e}")
            return image_path

    def _deskew_image(self, image):
        """Deskew the image."""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image

            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[-1]

            # Correct the angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            return rotated

        except Exception:
            return image

    def _convert_results_to_elements(self, results: List) -> List[ContentElement]:
        """
        Convert PaddleOCR results to content elements.

        Args:
            results: PaddleOCR detection results

        Returns:
            List of content elements
        """
        elements = []

        for line in results:
            if not line:
                continue

            bbox, (text, confidence) = line
            if not text.strip():
                continue

            # Clean text
            clean_text = text.strip()

            # Convert bounding box format
            flat_bbox = [coord for point in bbox for coord in point]
            x_coords = flat_bbox[::2]
            y_coords = flat_bbox[1::2]

            from ...core.document import BoundingBox
            bounding_box = BoundingBox(
                x0=min(x_coords),
                y0=min(y_coords),
                x1=max(x_coords),
                y1=max(y_coords)
            )

            # Detect content type
            content_type = self._detect_content_type(clean_text)

            element = ContentElement(
                text=clean_text,
                content_type=content_type,
                confidence=float(confidence),
                bounding_box=bounding_box,
                metadata={
                    'source': 'paddleocr',
                    'detection_method': 'deeplearning'
                }
            )

            elements.append(element)

        return elements

    def _detect_content_type(self, text: str) -> ContentType:
        """Detect content type from text."""
        import re

        # Check for table patterns
        if re.search(r'(\t|\s{3,}|,)\s*\w+', text) and text.count(' ') > 3:
            return ContentType.TABLE

        # Check for code patterns
        code_indicators = ['def ', 'function', 'var ', 'let ', 'const ', 'import ', 'if ', 'for ', 'while ']
        if any(indicator in text for indicator in code_indicators):
            return ContentType.CODE

        # Check for mathematical patterns
        math_patterns = [r'\$[^$]+\$', r'\\[a-zA-Z]+', r'[α-ωΑ-Ω]', r'[∑∫∏∂∇]']
        if any(re.search(pattern, text) for pattern in math_patterns):
            return ContentType.MATHEMATICAL

        return ContentType.TEXT
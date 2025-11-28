"""
EasyOCR engine implementation.

This module provides EasyOCR integration for VisionPDF,
offering deep learning-based OCR with support for 80+ languages.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import easyocr
    import cv2
    import numpy as np
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    cv2 = None
    np = None
    EASYOCR_AVAILABLE = False

from ..base import OCREngine, OCRResult, ContentElement, ContentType
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR engine implementation."""

    def __init__(self, config):
        """
        Initialize EasyOCR engine.

        Args:
            config: OCR configuration
        """
        self.reader = None
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh_sim', 'zh_tra',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no', 'fi', 'he'
        ]
        super().__init__(config)

    def _initialize_engine(self) -> None:
        """Initialize EasyOCR engine."""
        try:
            import easyocr

            # Convert language codes to EasyOCR format
            easyocr_langs = self._convert_languages(self.config.languages)

            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(
                easyocr_langs,
                gpu=False,  # Use CPU for compatibility
                download_enabled=True
            )

            logger.info(f"EasyOCR initialized with languages: {easyocr_langs}")

        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def _convert_languages(self, languages: List[str]) -> List[str]:
        """Convert language codes to EasyOCR format."""
        lang_mapping = {
            'eng': 'en',
            'spa': 'es',
            'fra': 'fr',
            'deu': 'de',
            'ita': 'it',
            'por': 'pt',
            'rus': 'ru',
            'jpn': 'ja',
            'kor': 'ko',
            'chi_sim': 'zh_sim',
            'chi_tra': 'zh_tra',
            'ara': 'ar',
            'hin': 'hi',
            'tha': 'th'
        }

        converted = []
        for lang in languages:
            easyocr_lang = lang_mapping.get(lang, lang)
            if easyocr_lang in self.supported_languages:
                converted.append(easyocr_lang)
            else:
                logger.warning(f"EasyOCR may not support language: {lang}")

        return converted if converted else ['en']  # Default to English

    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            return self.reader is not None
        except Exception:
            return False

    def extract_text_from_image(self, image_path: str) -> OCRResult:
        """
        Extract text from image using EasyOCR.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        if not self.reader:
            return OCRResult(text="", confidence=0.0, metadata={'error': 'EasyOCR not initialized'})

        try:
            # Preprocess image if requested
            processed_image_path = self._preprocess_image(image_path)

            # Extract text using EasyOCR
            results = self.reader.readtext(processed_image_path)

            if not results:
                return OCRResult(text="", confidence=0.0, metadata={'warning': 'No text detected'})

            # Process results
            text_parts = []
            confidences = []
            bounding_boxes = []

            for (bbox, text, confidence) in results:
                if text.strip():
                    text_parts.append(text.strip())
                    confidences.append(confidence)

                    # Convert bounding box format
                    # EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
                    'engine': 'easyocr',
                    'detection_count': len(results),
                    'languages_used': self.config.languages
                }
            )

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return OCRResult(text="", confidence=0.0, metadata={'error': str(e)})

    def extract_text_with_layout(self, image_path: str) -> List[ContentElement]:
        """
        Extract text with layout information.

        Args:
            image_path: Path to image file

        Returns:
            List of content elements with layout info
        """
        if not self.reader:
            return []

        try:
            # Preprocess image
            processed_image_path = self._preprocess_image(image_path)

            # Extract text with detailed information
            results = self.reader.readtext(processed_image_path)

            # Convert results to content elements
            elements = self._convert_results_to_elements(results)

            # Clean up temporary files
            if processed_image_path != image_path:
                try:
                    os.unlink(processed_image_path)
                except:
                    pass

            return elements

        except Exception as e:
            logger.error(f"EasyOCR layout analysis failed: {e}")
            return []

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results.

        Args:
            image_path: Path to original image

        Returns:
            Path to preprocessed image
        """
        if not self.config.preprocessing or not EASYOCR_AVAILABLE:
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
            logger.warning(f"EasyOCR image preprocessing failed: {e}")
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
        Convert EasyOCR results to content elements.

        Args:
            results: EasyOCR detection results

        Returns:
            List of content elements
        """
        elements = []

        for (bbox, text, confidence) in results:
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
                    'source': 'easyocr',
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
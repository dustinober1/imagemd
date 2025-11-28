"""
Tesseract OCR engine implementation.

This module provides Tesseract OCR integration for VisionPDF,
supporting multiple languages and advanced preprocessing.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    Image = None
    pytesseract = None
    CV2_AVAILABLE = False

from ..base import OCREngine, OCRResult, ContentElement, ContentType
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation."""

    def __init__(self, config):
        """
        Initialize Tesseract OCR engine.

        Args:
            config: OCR configuration
        """
        self.tesseract_path = None
        self.supported_languages = [
            'eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'jpn',
            'kor', 'chi_sim', 'chi_tra', 'ara', 'hin', 'tha'
        ]
        super().__init__(config)

    def _initialize_engine(self) -> None:
        """Initialize Tesseract engine."""
        try:
            # Check if tesseract is available
            import pytesseract

            # Try to find tesseract executable
            tesseract_cmd = os.environ.get('TESSERACT_CMD')
            if tesseract_cmd and os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self.tesseract_path = tesseract_cmd
            else:
                # Try common locations
                common_paths = [
                    '/usr/bin/tesseract',
                    '/usr/local/bin/tesseract',
                    '/opt/homebrew/bin/tesseract',
                    'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.tesseract_path = path
                        break

            # Test tesseract availability
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized (version: {version})")

        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text_from_image(self, image_path: str) -> OCRResult:
        """
        Extract text from image using Tesseract.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        try:
            import pytesseract
            from PIL import Image

            # Preprocess image if requested
            processed_image_path = self._preprocess_image(image_path)

            # Open image
            image = Image.open(processed_image_path)

            # Configure Tesseract
            lang_string = '+'.join(self.config.languages)
            config = self._get_tesseract_config()

            # Extract text with confidence data
            data = pytesseract.image_to_data(
                image,
                lang=lang_string,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            # Process results
            text_parts = []
            confidences = []
            bounding_boxes = []

            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Skip low confidence results
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(float(data['conf'][i]) / 100.0)

                        # Extract bounding box
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bounding_boxes.append((x, y, x + w, y + h))

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
                language=lang_string,
                bounding_boxes=bounding_boxes,
                metadata={
                    'engine': 'tesseract',
                    'word_count': len(text_parts),
                    'languages_used': self.config.languages
                }
            )

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return OCRResult(text="", confidence=0.0, metadata={'error': str(e)})

    def extract_text_with_layout(self, image_path: str) -> List[ContentElement]:
        """
        Extract text with layout information.

        Args:
            image_path: Path to image file

        Returns:
            List of content elements with layout info
        """
        try:
            import pytesseract
            from PIL import Image

            # Preprocess image
            processed_image_path = self._preprocess_image(image_path)

            # Open image
            image = Image.open(processed_image_path)

            # Configure Tesseract for layout analysis
            lang_string = '+'.join(self.config.languages)
            config = '--psm 6 '  # Assume uniform block of text

            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image,
                lang=lang_string,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            # Group text into blocks/paragraphs
            elements = self._group_text_into_elements(data)

            # Clean up temporary files
            if processed_image_path != image_path:
                try:
                    os.unlink(processed_image_path)
                except:
                    pass

            return elements

        except Exception as e:
            logger.error(f"Tesseract layout analysis failed: {e}")
            return []

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results.

        Args:
            image_path: Path to original image

        Returns:
            Path to preprocessed image
        """
        if not self.config.preprocessing or not CV2_AVAILABLE:
            return image_path

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return image_path

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)

            # Apply thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Deskew if requested
            if self.config.deskew:
                thresh = self._deskew_image(thresh)

            # Enhance contrast if requested
            if self.config.enhancement:
                thresh = self._enhance_contrast(thresh)

            # Save processed image to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(temp_file.name, thresh)

            return temp_file.name

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path

    def _deskew_image(self, image):
        """Deskew the image."""
        if not CV2_AVAILABLE or cv2 is None or np is None:
            return image

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

    def _enhance_contrast(self, image):
        """Enhance image contrast."""
        if not CV2_AVAILABLE or cv2 is None:
            return image

        try:
            # Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            return enhanced
        except Exception:
            return image

    def _get_tesseract_config(self) -> str:
        """Get Tesseract configuration string."""
        config_parts = []

        # Page segmentation mode
        config_parts.append('--psm 6')  # Assume uniform block of text

        # OEM (OCR Engine Mode)
        config_parts.append('--oem 3')  # Default

        # Additional options
        if self.config.preprocessing:
            config_parts.append('-c tessedit_do_invert=0')
            config_parts.append('-c tessedit_do_invert=0')

        return ' '.join(config_parts)

    def _group_text_into_elements(self, data: Dict[str, List]) -> List[ContentElement]:
        """
        Group OCR text data into content elements.

        Args:
            data: Tesseract OCR data dictionary

        Returns:
            List of content elements
        """
        elements = []

        # Group text by line numbers
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Skip low confidence results
                text = data['text'][i].strip()
                if text:
                    line_num = data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []

                    lines[line_num].append({
                        'text': text,
                        'confidence': float(data['conf'][i]) / 100.0,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    })

        # Create elements from lines
        for line_num, words in sorted(lines.items()):
            if not words:
                continue

            # Sort words by x position
            words.sort(key=lambda w: w['x'])

            # Combine words into line text
            line_text = ' '.join(word['text'] for word in words)
            avg_confidence = sum(word['confidence'] for word in words) / len(words)

            # Calculate bounding box for the entire line
            min_x = min(word['x'] for word in words)
            min_y = min(word['y'] for word in words)
            max_x = max(word['x'] + word['w'] for word in words)
            max_y = max(word['y'] + word['h'] for word in words)

            # Create bounding box object
            from ...core.document import BoundingBox
            bbox = BoundingBox(x0=min_x, y0=min_y, x1=max_x, y1=max_y)

            # Detect content type
            content_type = self._detect_content_type(line_text)

            element = ContentElement(
                text=line_text,
                content_type=content_type,
                confidence=avg_confidence,
                bounding_box=bbox,
                metadata={
                    'source': 'tesseract',
                    'line_number': line_num,
                    'word_count': len(words)
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
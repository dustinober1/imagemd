"""
OCR integration for VisionPDF.

This package provides OCR capabilities as a fallback when VLM processing fails,
supporting multiple OCR engines including Tesseract, EasyOCR, and PaddleOCR.
"""

from .base import (
    OCRResult,
    OCRConfig,
    OCREngine,
    OCRFallbackManager,
    OCRPostProcessor
)

# Import engine implementations
try:
    from .engines.tesseract_engine import TesseractEngine
except ImportError:
    TesseractEngine = None

try:
    from .engines.easyocr_engine import EasyOCREngine
except ImportError:
    EasyOCREngine = None

try:
    from .engines.paddleocr_engine import PaddleOCREngine
except ImportError:
    PaddleOCREngine = None

__all__ = [
    'OCRResult',
    'OCRConfig',
    'OCREngine',
    'OCRFallbackManager',
    'OCRPostProcessor',
    'TesseractEngine',
    'EasyOCREngine',
    'PaddleOCREngine'
]


def create_ocr_manager(config: OCRConfig = None) -> OCRFallbackManager:
    """
    Create and configure OCR fallback manager.

    Args:
        config: OCR configuration (optional)

    Returns:
        Configured OCR fallback manager
    """
    if config is None:
        config = OCRConfig()

    return OCRFallbackManager(config)


def get_available_engines() -> list:
    """
    Get list of available OCR engines.

    Returns:
        List of available engine names
    """
    engines = []

    if TesseractEngine:
        try:
            engine = TesseractEngine(OCRConfig())
            if engine.is_available():
                engines.append('tesseract')
        except:
            pass

    if EasyOCREngine:
        try:
            engine = EasyOCREngine(OCRConfig())
            if engine.is_available():
                engines.append('easyocr')
        except:
            pass

    if PaddleOCREngine:
        try:
            engine = PaddleOCREngine(OCRConfig())
            if engine.is_available():
                engines.append('paddleocr')
        except:
            pass

    return engines
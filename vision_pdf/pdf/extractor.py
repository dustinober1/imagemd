"""
PDF text and content extraction utilities.

This module provides tools for extracting text, images, and other content
from PDF documents using various libraries and techniques.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import io
import logging

from ..core.document import Page, ContentElement, BoundingBox, ContentType
from ..config.settings import VisionPDFConfig

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    PDF content extractor with multiple extraction methods.

    This class provides various methods for extracting text, images,
    and structured content from PDF documents.
    """

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the PDF extractor.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self.ocr_enabled = config.ocr.enabled

    def extract_text_from_page(
        self,
        pdf_path: Path,
        page_number: int,
        method: str = "auto"
    ) -> str:
        """
        Extract text from a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)
            method: Extraction method ("pymupdf", "pdfplumber", "auto")

        Returns:
            Extracted text as string
        """
        if method == "auto":
            # Try both methods and combine results
            text_pymupdf = self._extract_text_pymupdf(pdf_path, page_number)
            text_pdfplumber = self._extract_text_pdfplumber(pdf_path, page_number)

            # Choose the longer, more complete text
            if len(text_pdfplumber) > len(text_pymupdf):
                return text_pdfplumber
            else:
                return text_pymupdf

        elif method == "pymupdf":
            return self._extract_text_pymupdf(pdf_path, page_number)

        elif method == "pdfplumber":
            return self._extract_text_pdfplumber(pdf_path, page_number)

        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def _extract_text_pymupdf(self, pdf_path: Path, page_number: int) -> str:
        """
        Extract text using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(str(pdf_path))
            try:
                if page_number < 0 or page_number >= len(doc):
                    raise ValueError(f"Invalid page number: {page_number}")

                page = doc[page_number]
                text = page.get_text()

                logger.debug(f"PyMuPDF extracted {len(text)} characters from page {page_number}")
                return text

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF text extraction failed for page {page_number}: {e}")
            return ""

    def _extract_text_pdfplumber(self, pdf_path: Path, page_number: int) -> str:
        """
        Extract text using pdfplumber.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            Extracted text
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number >= len(pdf.pages):
                    raise ValueError(f"Invalid page number: {page_number}")

                page = pdf.pages[page_number]
                text = page.extract_text()

                logger.debug(f"pdfplumber extracted {len(text)} characters from page {page_number}")
                return text

        except Exception as e:
            logger.error(f"pdfplumber text extraction failed for page {page_number}: {e}")
            return ""

    def extract_text_with_layout(
        self,
        pdf_path: Path,
        page_number: int
    ) -> List[ContentElement]:
        """
        Extract text with layout information.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            List of ContentElement objects with layout information
        """
        elements = []

        try:
            # Use PyMuPDF for detailed text extraction with layout
            doc = fitz.open(str(pdf_path))
            try:
                page = doc[page_number]
                text_dict = page.get_text("dict")

                for block in text_dict.get('blocks', []):
                    if 'lines' in block:
                        # Extract text with position information
                        bbox = block['bbox']
                        bounding_box = BoundingBox(
                            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                            page=page_number
                        )

                        # Combine all text in the block
                        block_text = ""
                        for line in block['lines']:
                            line_text = ""
                            for span in line.get('spans', []):
                                line_text += span.get('text', '')
                            block_text += line_text + "\n"

                        block_text = block_text.strip()
                        if block_text:
                            element = ContentElement(
                                content_type=ContentType.TEXT,
                                text=block_text,
                                confidence=0.9,  # High confidence for direct extraction
                                bounding_box=bounding_box,
                                metadata={
                                    'source': 'pymupdf',
                                    'font_size': self._get_average_font_size(block)
                                }
                            )
                            elements.append(element)

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"Layout text extraction failed for page {page_number}: {e}")

        return elements

    def extract_images_from_page(
        self,
        pdf_path: Path,
        page_number: int,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Extract images from a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)
            output_dir: Directory to save extracted images

        Returns:
            List of paths to extracted image files
        """
        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []

        try:
            doc = fitz.open(str(pdf_path))
            try:
                page = doc[page_number]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Skip CMYK images (not supported by PIL)
                    if pix.n - pix.alpha < 4:
                        img_path = output_dir / f"page_{page_number}_img_{img_index}.png"
                        pix.save(img_path)
                        image_paths.append(img_path)

                    pix = pix  # Release memory

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"Image extraction failed for page {page_number}: {e}")

        logger.debug(f"Extracted {len(image_paths)} images from page {page_number}")
        return image_paths

    def extract_tables_from_page(
        self,
        pdf_path: Path,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            List of table data dictionaries
        """
        tables = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < len(pdf.pages):
                    page = pdf.pages[page_number]
                    table_data = page.extract_tables()

                    for table_index, table in enumerate(table_data):
                        if table and len(table) > 1:
                            table_dict = {
                                'index': table_index,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0,
                                'data': table,
                                'bbox': None  # pdfplumber doesn't provide table bounding boxes
                            }
                            tables.append(table_dict)

        except Exception as e:
            logger.error(f"Table extraction failed for page {page_number}: {e}")

        logger.debug(f"Extracted {len(tables)} tables from page {page_number}")
        return tables

    def extract_annotations_from_page(
        self,
        pdf_path: Path,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Extract annotations from a specific page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            List of annotation dictionaries
        """
        annotations = []

        try:
            doc = fitz.open(str(pdf_path))
            try:
                page = doc[page_number]
                page_annot = page.annots()

                for annot in page_annot:
                    annot_dict = {
                        'type': annot.type[1],
                        'content': annot.info.get("content", ""),
                        'author': annot.info.get("title", ""),
                        'created': annot.info.get("creationDate", ""),
                        'modified': annot.info.get("modDate", ""),
                        'bbox': list(annot.rect),
                        'color': annot.colors.get("stroke", None)
                    }
                    annotations.append(annot_dict)

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"Annotation extraction failed for page {page_number}: {e}")

        logger.debug(f"Extracted {len(annotations)} annotations from page {page_number}")
        return annotations

    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        try:
            doc = fitz.open(str(pdf_path))
            try:
                # Basic metadata
                metadata.update(doc.metadata)

                # Additional metadata
                metadata.update({
                    'page_count': len(doc),
                    'is_encrypted': doc.is_encrypted,
                    'is_pdf': True,
                    'pdf_version': doc.metadata.get('format', 'Unknown').replace('PDF ', ''),
                    'permissions': doc.permissions,
                })

                # Page size information
                if len(doc) > 0:
                    first_page = doc[0]
                    rect = first_page.rect
                    metadata['page_size'] = {
                        'width': rect.width,
                        'height': rect.height
                    }

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"Metadata extraction failed for {pdf_path}: {e}")

        return metadata

    def extract_text_with_ocr(
        self,
        pdf_path: Path,
        page_number: int,
        language: Optional[str] = None
    ) -> str:
        """
        Extract text using OCR (if available).

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)
            language: Language code for OCR (uses config default if None)

        Returns:
            OCR-extracted text
        """
        if not self.ocr_enabled:
            logger.warning("OCR is not enabled in configuration")
            return ""

        if language is None:
            language = self.config.ocr.languages[0] if self.config.ocr.languages else "en"

        try:
            # First, render page to image
            doc = fitz.open(str(pdf_path))
            try:
                page = doc[page_number]

                # Render page to image at high DPI for OCR
                zoom = self.config.processing.dpi / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)

                # Convert to PIL Image for OCR
                img_data = pix.tobytes("ppm")
                from PIL import Image
                image = Image.open(io.BytesIO(img_data))

                # Perform OCR using configured engine
                ocr_text = self._perform_ocr(image, language)

                logger.debug(f"OCR extracted {len(ocr_text)} characters from page {page_number}")
                return ocr_text

            finally:
                doc.close()

        except Exception as e:
            logger.error(f"OCR text extraction failed for page {page_number}: {e}")
            return ""

    def _perform_ocr(self, image, language: str) -> str:
        """
        Perform OCR on an image using the configured engine.

        Args:
            image: PIL Image object
            language: Language code

        Returns:
            OCR-extracted text
        """
        if self.config.ocr.engine.lower() == "easyocr":
            return self._ocr_with_easyocr(image, language)
        elif self.config.ocr.engine.lower() == "tesseract":
            return self._ocr_with_tesseract(image, language)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.config.ocr.engine}")

    def _ocr_with_easyocr(self, image, language: str) -> str:
        """
        Perform OCR using EasyOCR.

        Args:
            image: PIL Image object
            language: Language code

        Returns:
            OCR-extracted text
        """
        try:
            import easyocr

            # Initialize reader (lazy initialization)
            if not hasattr(self, '_easyocr_reader'):
                self._easyocr_reader = easyocr.Reader(
                    self.config.ocr.languages,
                    gpu=False  # Use CPU by default for compatibility
                )

            # Perform OCR
            results = self._easyocr_reader.readtext(image)

            # Extract text from results
            text_lines = [result[1] for result in results if result[2] > self.config.ocr.confidence_threshold]
            return "\n".join(text_lines)

        except ImportError:
            logger.error("EasyOCR is not installed. Install with: pip install easyocr")
            return ""
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""

    def _ocr_with_tesseract(self, image, language: str) -> str:
        """
        Perform OCR using Tesseract.

        Args:
            image: PIL Image object
            language: Language code

        Returns:
            OCR-extracted text
        """
        try:
            import pytesseract

            # Configure Tesseract
            config = f'--oem 3 --psm 6 -l {language}'

            # Perform OCR
            text = pytesseract.image_to_string(image, config=config)

            return text.strip()

        except ImportError:
            logger.error("Tesseract is not installed. Install with: pip install pytesseract")
            return ""
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def _get_average_font_size(self, block: Dict[str, Any]) -> float:
        """
        Get average font size from a text block.

        Args:
            block: Text block dictionary

        Returns:
            Average font size
        """
        try:
            sizes = []
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    size = span.get('size')
                    if size:
                        sizes.append(size)

            return sum(sizes) / len(sizes) if sizes else 12.0

        except Exception:
            return 12.0

    def get_extraction_quality_score(
        self,
        pdf_path: Path,
        page_number: int
    ) -> float:
        """
        Get a quality score for text extraction on a page.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Extract text using multiple methods
            text_pymupdf = self._extract_text_pymupdf(pdf_path, page_number)
            text_pdfplumber = self._extract_text_pdfplumber(pdf_path, page_number)

            if not text_pymupdf and not text_pdfplumber:
                return 0.0

            # Calculate quality metrics
            length_score = min(len(max(text_pymupdf, text_pdfplumber)) / 1000, 1.0)

            # Check for common extraction artifacts
            artifacts = ['\x00', '\x0c', '\x14']  # Common PDF artifacts
            artifact_penalty = sum(text_pymupdf.count(art) + text_pdfplumber.count(art) for art in artifacts) / 100

            # Consistency between methods
            if text_pymupdf and text_pdfplumber:
                consistency_score = 1.0 - abs(len(text_pymupdf) - len(text_pdfplumber)) / max(len(text_pymupdf), len(text_pdfplumber))
            else:
                consistency_score = 0.5

            quality_score = (length_score + consistency_score - artifact_penalty) / 2
            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"Quality assessment failed for page {page_number}: {e}")
            return 0.0
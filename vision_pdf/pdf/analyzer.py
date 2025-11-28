"""
PDF document analysis and layout detection.

This module provides tools for analyzing PDF document structure,
detecting content types, and extracting metadata.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import logging

from ..core.document import Document, Page, ContentElement, BoundingBox, ContentType
from ..config.settings import VisionPDFConfig

logger = logging.getLogger(__name__)


class PDFAnalyzer:
    """
    PDF document analyzer for layout and content detection.

    This class analyzes PDF documents to detect structure, content types,
    and extract metadata for processing optimization.
    """

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the PDF analyzer.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config

    def analyze_document(self, pdf_path: Path) -> Document:
        """
        Analyze a PDF document and extract metadata and structure.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Document object with analysis results
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Use PyMuPDF for metadata and basic analysis
        fitz_doc = fitz.open(str(pdf_path))
        try:
            # Extract metadata
            metadata = self._extract_metadata(fitz_doc)

            # Analyze document structure
            document = Document(
                file_path=pdf_path,
                title=metadata.get('title'),
                author=metadata.get('author'),
                creator=metadata.get('creator'),
                producer=metadata.get('producer'),
                creation_date=metadata.get('creation_date'),
                modification_date=metadata.get('modification_date'),
                page_count=len(fitz_doc),
                metadata=metadata
            )

            # Analyze each page
            for page_num in range(len(fitz_doc)):
                page = self._analyze_page(fitz_doc, page_num)
                document.add_page(page)

            # Analyze overall document characteristics
            document.metadata.update(self._analyze_document_characteristics(document))

            logger.info(f"Analyzed document: {pdf_path} ({document.page_count} pages)")
            return document

        finally:
            fitz_doc.close()

    def _analyze_page(self, fitz_doc: fitz.Document, page_number: int) -> Page:
        """
        Analyze a single page for structure and content.

        Args:
            fitz_doc: PyMuPDF document object
            page_number: Page number (0-based)

        Returns:
            Page object with analysis results
        """
        page = fitz_doc[page_number]
        rect = page.rect

        # Create Page object
        page_obj = Page(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            dpi=self.config.processing.dpi,
            metadata={}
        )

        # Extract text and analyze structure
        text_dict = page.get_text("dict")
        self._analyze_text_blocks(text_dict, page_obj)

        # Use pdfplumber for more detailed analysis
        self._analyze_with_pdfplumber(fitz_doc, page_number, page_obj)

        # Detect tables
        self._detect_tables(fitz_doc, page_number, page_obj)

        # Detect mathematical expressions
        self._detect_mathematical_expressions(page_obj)

        # Detect code blocks
        self._detect_code_blocks(page_obj)

        logger.debug(f"Analyzed page {page_number}: {len(page_obj.elements)} elements detected")
        return page_obj

    def _extract_metadata(self, fitz_doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.

        Args:
            fitz_doc: PyMuPDF document object

        Returns:
            Dictionary with metadata
        """
        metadata = fitz_doc.metadata
        return {
            'title': metadata.get('title'),
            'author': metadata.get('author'),
            'subject': metadata.get('subject'),
            'creator': metadata.get('creator'),
            'producer': metadata.get('producer'),
            'creation_date': metadata.get('creationDate'),
            'modification_date': metadata.get('modDate'),
            'encrypted': fitz_doc.is_encrypted,
            'pdf_version': fitz_doc.pdf_version(),
            'page_count': len(fitz_doc)
        }

    def _analyze_text_blocks(self, text_dict: Dict[str, Any], page_obj: Page) -> None:
        """
        Analyze text blocks from PyMuPDF text extraction.

        Args:
            text_dict: Text dictionary from PyMuPDF
            page_obj: Page object to add elements to
        """
        for block in text_dict.get('blocks', []):
            if 'lines' in block:
                # This is a text block
                bbox = block['bbox']
                bounding_box = BoundingBox(
                    x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], page=page_obj.page_number
                )

                # Extract text from lines
                text_lines = []
                for line in block['lines']:
                    line_text = ''
                    for span in line.get('spans', []):
                        line_text += span.get('text', '')
                    if line_text.strip():
                        text_lines.append(line_text)

                if text_lines:
                    full_text = '\n'.join(text_lines)

                    # Classify content type
                    content_type = self._classify_text_content(full_text)

                    # Create content element
                    element = ContentElement(
                        content_type=content_type,
                        text=full_text,
                        confidence=self._estimate_text_confidence(full_text),
                        bounding_box=bounding_box,
                        metadata={
                            'font_size': self._estimate_font_size(block),
                            'lines_count': len(text_lines),
                            'source': 'pymupdf'
                        }
                    )

                    page_obj.add_element(element)

    def _analyze_with_pdfplumber(self, fitz_doc: fitz.Document, page_number: int, page_obj: Page) -> None:
        """
        Analyze page using pdfplumber for enhanced table and structure detection.

        Args:
            fitz_doc: PyMuPDF document object
            page_number: Page number
            page_obj: Page object to update
        """
        try:
            # Reopen with pdfplumber for analysis
            pdf_path = page_obj.metadata.get('pdf_path')
            if pdf_path:
                with pdfplumber.open(pdf_path) as pdf:
                    if page_number < len(pdf.pages):
                        plumber_page = pdf.pages[page_number]

                        # Extract text with position information
                        text = plumber_page.extract_text()
                        if text:
                            page_obj.raw_text = text

                        # Detect characters for better analysis
                        chars = plumber_page.chars
                        if chars:
                            self._analyze_characters(chars, page_obj)

        except Exception as e:
            logger.debug(f"pdfplumber analysis failed for page {page_number}: {e}")

    def _detect_tables(self, fitz_doc: fitz.Document, page_number: int, page_obj: Page) -> None:
        """
        Detect tables on the page.

        Args:
            fitz_doc: PyMuPDF document object
            page_number: Page number
            page_obj: Page object to update
        """
        try:
            # Use pdfplumber for table detection
            pdf_path = page_obj.metadata.get('pdf_path')
            if not pdf_path:
                return

            with pdfplumber.open(pdf_path) as pdf:
                if page_number < len(pdf.pages):
                    plumber_page = pdf.pages[page_number]
                    tables = plumber_page.extract_tables()

                    for table_data in tables:
                        if table_data and len(table_data) > 1:
                            # Convert table to markdown-like text
                            table_text = self._table_to_text(table_data)

                            # Create bounding box (estimate from page size)
                            bounding_box = BoundingBox(
                                x0=50, y0=50, x1=page_obj.width - 50, y1=page_obj.height - 50,
                                page=page_obj.page_number
                            )

                            element = ContentElement(
                                content_type=ContentType.TABLE,
                                text=table_text,
                                confidence=0.7,  # Table detection confidence
                                bounding_box=bounding_box,
                                metadata={
                                    'rows': len(table_data),
                                    'cols': len(table_data[0]) if table_data else 0,
                                    'source': 'pdfplumber'
                                }
                            )

                            page_obj.add_element(element)

        except Exception as e:
            logger.debug(f"Table detection failed for page {page_number}: {e}")

    def _detect_mathematical_expressions(self, page_obj: Page) -> None:
        """
        Detect mathematical expressions in page text.

        Args:
            page_obj: Page object to analyze
        """
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\[.*?\\\]',  # LaTeX display math
            r'\\\(.*?\\\)',  # LaTeX inline math
            r'\w+\s*[=≠≤≥<>]\s*[\w\d\(\)\+\-\*/\^]+',  # Mathematical expressions
            r'∫.*?dx',  # Integrals
            r'∑.*?=.*',  # Summations
            r'∂f/∂[a-zA-Z]',  # Partial derivatives
        ]

        text_elements = page_obj.get_text_elements()
        for element in text_elements:
            text = element.text
            detected_expressions = []

            for pattern in math_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
                for match in matches:
                    detected_expressions.append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })

            if detected_expressions:
                # Update element metadata
                element.metadata['math_expressions'] = detected_expressions
                element.metadata['has_math'] = True

                # If significant math content, classify as mathematical
                if len(detected_expressions) > 2:
                    element.content_type = ContentType.MATHEMATICAL

    def _detect_code_blocks(self, page_obj: Page) -> None:
        """
        Detect code blocks in page text.

        Args:
            page_obj: Page object to analyze
        """
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python function definitions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'#!/',  # Shebang lines
            r'<\?php',  # PHP tags
            r'<script',  # Script tags
            r'{\s*\n.*?\n\s*}',  # Code blocks with braces
        ]

        text_elements = page_obj.get_text_elements()
        for element in text_elements:
            text = element.text
            code_score = 0

            for pattern in code_indicators:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    code_score += 1

            # Check for code-like characteristics
            if re.search(r'^\s*[}>\]\)]\s*$', text, re.MULTILINE):  # Lines ending with closing brackets
                code_score += 1

            if 'for (' in text or 'while (' in text or 'if (' in text:
                code_score += 1

            # If high code score, classify as code
            if code_score >= 2:
                element.content_type = ContentType.CODE
                element.metadata['code_score'] = code_score

    def _classify_text_content(self, text: str) -> ContentType:
        """
        Classify text content based on patterns and characteristics.

        Args:
            text: Text to classify

        Returns:
            Detected content type
        """
        text_lower = text.lower().strip()

        # Check for headings
        if len(text) < 100 and (
            re.match(r'^#{1,6}\s+', text) or  # Markdown headers
            (len(text.split()) < 10 and text.isupper()) or  # Short, all caps
            (re.match(r'^[A-Z][a-z\s]+:$', text))  # Title-like
        ):
            return ContentType.TEXT

        # Check for lists
        if re.match(r'^\s*[-*+]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return ContentType.TEXT

        # Check for table-like content
        if '|' in text and text.count('|') >= 3:
            return ContentType.TABLE

        # Default to text
        return ContentType.TEXT

    def _estimate_text_confidence(self, text: str) -> float:
        """
        Estimate confidence level for extracted text.

        Args:
            text: Extracted text

        Returns:
            Confidence score between 0 and 1
        """
        if not text or not text.strip():
            return 0.0

        # Base confidence
        confidence = 0.8

        # Adjust based on text characteristics
        if text.isupper() and len(text) > 50:
            confidence -= 0.1  # Might be shouting/headers

        if len(text) < 10:
            confidence -= 0.2  # Very short text

        # Check for common OCR errors
        ocr_error_patterns = [r'[l1iI]+', r'[0oO]+', r'[rn]+']
        for pattern in ocr_error_patterns:
            if re.search(pattern, text):
                confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _estimate_font_size(self, block: Dict[str, Any]) -> float:
        """
        Estimate font size from text block.

        Args:
            block: Text block dictionary from PyMuPDF

        Returns:
            Estimated font size
        """
        try:
            spans = []
            for line in block.get('lines', []):
                spans.extend(line.get('spans', []))

            if spans:
                # Average font size across all spans
                font_sizes = [span.get('size', 12) for span in spans if span.get('size')]
                return sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

        except Exception:
            pass

        return 12.0  # Default font size

    def _analyze_characters(self, chars: List[Dict[str, Any]], page_obj: Page) -> None:
        """
        Analyze character-level information for better understanding.

        Args:
            chars: List of character dictionaries from pdfplumber
            page_obj: Page object to update
        """
        if not chars:
            return

        # Character statistics
        font_sizes = [char.get('size', 12) for char in chars if char.get('size')]
        if font_sizes:
            page_obj.metadata['avg_font_size'] = sum(font_sizes) / len(font_sizes)
            page_obj.metadata['max_font_size'] = max(font_sizes)
            page_obj.metadata['min_font_size'] = min(font_sizes)

        # Character count
        page_obj.metadata['char_count'] = len(chars)

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to text representation.

        Args:
            table_data: Table data as list of rows

        Returns:
            Text representation of table
        """
        if not table_data:
            return ""

        text_rows = []
        for row in table_data:
            if row:
                # Clean cell values and join
                clean_row = [str(cell) if cell is not None else "" for cell in row]
                text_rows.append(" | ".join(clean_row))

        return "\n".join(text_rows)

    def _analyze_document_characteristics(self, document: Document) -> Dict[str, Any]:
        """
        Analyze overall document characteristics.

        Args:
            document: Document object

        Returns:
            Dictionary with document characteristics
        """
        characteristics = {}

        # Page count classification
        if document.page_count <= 5:
            characteristics['length_class'] = 'short'
        elif document.page_count <= 20:
            characteristics['length_class'] = 'medium'
        else:
            characteristics['length_class'] = 'long'

        # Content type analysis
        total_elements = document.get_total_elements()
        table_count = len(document.get_all_elements_by_type(ContentType.TABLE))
        math_count = len(document.get_all_elements_by_type(ContentType.MATHEMATICAL))
        code_count = len(document.get_all_elements_by_type(ContentType.CODE))

        characteristics['total_elements'] = total_elements
        characteristics['table_count'] = table_count
        characteristics['math_count'] = math_count
        characteristics['code_count'] = code_count

        # Determine document type
        if math_count > total_elements * 0.1:
            characteristics['document_type'] = 'academic/technical'
        elif code_count > total_elements * 0.1:
            characteristics['document_type'] = 'programming/documentation'
        elif table_count > total_elements * 0.2:
            characteristics['document_type'] = 'data/business'
        else:
            characteristics['document_type'] = 'general'

        return characteristics
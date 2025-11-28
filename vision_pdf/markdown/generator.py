"""
Core markdown generation system for VisionPDF.

This module provides the main markdown generator that coordinates
formatting of processed PDF content into structured markdown.
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from ..core.document import Document, Page, ContentElement, ContentType
from ..config.settings import VisionPDFConfig, ProcessingMode
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MarkdownGenerator:
    """
    Main markdown generator for PDF to markdown conversion.

    This class orchestrates the conversion of processed PDF content
    into well-formatted markdown documents.
    """

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the markdown generator.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self._init_formatters()

    def _init_formatters(self) -> None:
        """Initialize specialized formatters for different content types."""
        from .formatters.tables import format_table_element, detect_and_format_tables
        from .formatters.math import format_math_element, detect_and_format_math
        from .formatters.code import detect_and_format_code

        # Import formatter functions
        self.format_table_element = format_table_element
        self.detect_and_format_tables = detect_and_format_tables
        self.format_math_element = format_math_element
        self.detect_and_format_math = detect_and_format_math
        self.detect_and_format_code = detect_and_format_code

    def generate_markdown(
        self,
        document: Document,
        processing_mode: ProcessingMode = ProcessingMode.HYBRID
    ) -> str:
        """
        Generate markdown from a processed document.

        Args:
            document: Processed document object
            processing_mode: Processing mode used for conversion

        Returns:
            Formatted markdown content
        """
        markdown_parts = []

        # Add document header
        markdown_parts.append(self._generate_document_header(document))

        # Process each page
        for page in document.pages:
            page_markdown = self._generate_page_content(page, processing_mode)
            markdown_parts.append(page_markdown)

        # Combine all parts
        markdown_content = "\n\n".join(markdown_parts)

        # Post-process markdown
        markdown_content = self._post_process_markdown(markdown_content)

        logger.info(f"Generated markdown for {document.page_count} pages")
        return markdown_content

    def _generate_document_header(self, document: Document) -> str:
        """
        Generate document header with metadata.

        Args:
            document: Document object

        Returns:
            Document header in markdown format
        """
        header_parts = []

        # Title
        title = document.title or document.file_path.stem
        header_parts.append(f"# {title}")

        # Metadata
        if document.author or document.creation_date:
            header_parts.append("## Document Information")
            if document.author:
                header_parts.append(f"**Author:** {document.author}")
            if document.creation_date:
                header_parts.append(f"**Created:** {document.creation_date}")
            if document.page_count:
                header_parts.append(f"**Pages:** {document.page_count}")

        return "\n\n".join(header_parts)

    def _generate_page_content(
        self,
        page: Page,
        processing_mode: ProcessingMode
    ) -> str:
        """
        Generate markdown content for a single page.

        Args:
            page: Page object
            processing_mode: Processing mode used

        Returns:
            Page content in markdown format
        """
        page_parts = []

        # Page header
        page_parts.append(f"## Page {page.page_number + 1}")

        if processing_mode == ProcessingMode.TEXT_ONLY:
            # Text-only processing
            page_parts.append(self._generate_text_only_content(page))
        else:
            # Vision or hybrid processing
            page_parts.append(self._generate_structured_content(page))

        return "\n\n".join(page_parts)

    def _generate_text_only_content(self, page: Page) -> str:
        """
        Generate content for text-only processing mode.

        Args:
            page: Page object

        Returns:
            Simple text content
        """
        if page.raw_text:
            return page.raw_text
        else:
            # Combine all text elements
            text_elements = page.get_text_elements()
            return "\n\n".join(element.text for element in text_elements)

    def _generate_structured_content(self, page: Page) -> str:
        """
        Generate structured content preserving layout.

        Args:
            page: Page object

        Returns:
            Structured markdown content
        """
        content_parts = []

        # Sort elements by bounding box (top to bottom, left to right)
        sorted_elements = self._sort_elements_by_position(page.elements)

        current_section = []

        for element in sorted_elements:
            if element.content_type == ContentType.TABLE:
                # Flush current section
                if current_section:
                    content_parts.append("\n".join(current_section))
                    current_section = []

                # Add table
                table_markdown = self._format_table(element)
                content_parts.append(table_markdown)

            elif element.content_type == ContentType.MATHEMATICAL:
                # Handle mathematical expressions
                math_markdown = self._format_mathematical(element)
                current_section.append(math_markdown)

            elif element.content_type == ContentType.CODE:
                # Handle code blocks
                code_markdown = self._format_code(element)
                current_section.append(code_markdown)

            else:
                # Regular text content
                text_markdown = self._format_text(element)
                current_section.append(text_markdown)

        # Flush remaining content
        if current_section:
            content_parts.append("\n".join(current_section))

        return "\n\n".join(content_parts)

    def _sort_elements_by_position(self, elements: List[ContentElement]) -> List[ContentElement]:
        """
        Sort content elements by their position on the page.

        Args:
            elements: List of content elements

        Returns:
            Elements sorted by position (top to bottom, left to right)
        """
        def position_key(element):
            if element.bounding_box:
                # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
                return (element.bounding_box.y0, element.bounding_box.x0)
            else:
                # Elements without bounding box go to the end
                return (float('inf'), float('inf'))

        return sorted(elements, key=position_key)

    def _format_table(self, element: ContentElement) -> str:
        """
        Format a table element as markdown.

        Args:
            element: Table content element

        Returns:
            Formatted table markdown
        """
        try:
            return self.format_table_element(element)
        except Exception as e:
            logger.warning(f"Failed to format table: {e}")
            return f"## Table\n\n{element.text}"

    def _format_mathematical(self, element: ContentElement) -> str:
        """
        Format a mathematical expression element.

        Args:
            element: Mathematical content element

        Returns:
            Formatted mathematical expression in LaTeX
        """
        try:
            return self.format_math_element(element)
        except Exception as e:
            logger.warning(f"Failed to format math: {e}")
            # Fallback to basic LaTeX formatting
            return f"\\[ {element.text} \\]"

    def _format_code(self, element: ContentElement) -> str:
        """
        Format a code block element.

        Args:
            element: Code content element

        Returns:
            Formatted code block with syntax highlighting
        """
        try:
            from .formatters.code import format_code_block
            # Create a temporary code block for formatting
            from .formatters.code import CodeBlock, CodeLanguage
            code_block = CodeBlock(
                lines=element.text.split('\n'),
                language=CodeLanguage.UNKNOWN,
                confidence=0.7,
                start_line=0,
                end_line=len(element.text.split('\n')) - 1
            )
            return format_code_block(code_block)
        except Exception as e:
            logger.warning(f"Failed to format code: {e}")
            # Fallback to generic code block
            return f"```\n{element.text}\n```"

    def _format_text(self, element: ContentElement) -> str:
        """
        Format a text content element.

        Args:
            element: Text content element

        Returns:
            Formatted text with appropriate markdown
        """
        text = element.text.strip()

        # Add heading formatting based on font size or metadata
        font_size = element.metadata.get('font_size', 12)

        if font_size > 16:
            # Likely a heading
            return f"### {text}"
        elif font_size > 14:
            # Likely a subheading
            return f"#### {text}"
        else:
            # Regular text
            return text

    def _post_process_markdown(self, markdown: str) -> str:
        """
        Post-process generated markdown for quality and consistency.

        Args:
            markdown: Raw generated markdown

        Returns:
            Improved markdown content
        """
        # Apply advanced formatting
        try:
            # Format any remaining tables
            markdown = self.detect_and_format_tables(markdown)

            # Format mathematical expressions
            markdown = self.detect_and_format_math(markdown)

            # Format code blocks
            markdown = self.detect_and_format_code(markdown)
        except Exception as e:
            logger.warning(f"Advanced formatting failed: {e}")

        # Remove excessive blank lines
        markdown = self._cleanup_blank_lines(markdown)

        # Fix common formatting issues
        markdown = self._fix_formatting_issues(markdown)

        # Ensure proper markdown structure
        markdown = self._ensure_proper_structure(markdown)

        return markdown

    def _cleanup_blank_lines(self, text: str) -> str:
        """Remove excessive blank lines."""
        import re
        # Replace multiple blank lines with at most 2
        return re.sub(r'\n{3,}', '\n\n', text)

    def _fix_formatting_issues(self, text: str) -> str:
        """Fix common markdown formatting issues."""
        # Fix spaces around headings
        import re
        text = re.sub(r'\n(#+)\s+', r'\n\1 ', text)

        # Fix trailing spaces
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        return '\n'.join(lines)

    def _ensure_proper_structure(self, text: str) -> str:
        """Ensure proper markdown document structure."""
        # Ensure document starts with a heading
        if not text.strip().startswith('#'):
            text = f"# Document\n\n{text}"

        # Ensure document ends without excessive newlines
        text = text.rstrip() + '\n'

        return text

    def generate_page_summary(self, page: Page) -> Dict[str, Any]:
        """
        Generate a summary of page processing results.

        Args:
            page: Processed page object

        Returns:
            Dictionary with page summary statistics
        """
        return {
            'page_number': page.page_number + 1,
            'total_elements': len(page.elements),
            'text_elements': len(page.get_text_elements()),
            'table_elements': len(page.get_table_elements()),
            'math_elements': len(page.get_mathematical_elements()),
            'code_elements': len(page.get_code_elements()),
            'has_raw_text': bool(page.raw_text),
            'processing_confidence': self._calculate_page_confidence(page)
        }

    def _calculate_page_confidence(self, page: Page) -> float:
        """
        Calculate overall confidence in page processing.

        Args:
            page: Page object

        Returns:
            Confidence score between 0 and 1
        """
        if not page.elements:
            return 0.0

        # Weight content types differently
        weights = {
            ContentType.TEXT: 0.3,
            ContentType.TABLE: 0.25,
            ContentType.MATHEMATICAL: 0.2,
            ContentType.CODE: 0.15,
            ContentType.IMAGE: 0.1
        }

        total_weighted_confidence = 0.0
        total_weight = 0.0

        for element in page.elements:
            weight = weights.get(element.content_type, 0.1)
            total_weighted_confidence += element.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_confidence / total_weight
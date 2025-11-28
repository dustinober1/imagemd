"""
Validation utilities for VisionPDF.

This module provides input validation, output validation, and data
integrity checks for the PDF processing pipeline.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import mimetypes
import logging
import os

from ..core.document import Document, Page, ContentElement
from ..config.settings import VisionPDFConfig
from ..utils.exceptions import ValidationError, PDFProcessingError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validator for input files and parameters."""

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the input validator.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self.max_file_size = config.max_file_size
        self.supported_formats = ['.pdf']

    def validate_pdf_file(self, pdf_path: Union[str, Path]) -> Path:
        """
        Validate that a file is a valid PDF file.

        Args:
            pdf_path: Path to the file to validate

        Returns:
            Validated Path object

        Raises:
            ValidationError: If file is invalid
        """
        pdf_path = Path(pdf_path)

        # Check if file exists
        if not pdf_path.exists():
            raise ValidationError(f"File not found: {pdf_path}")

        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise ValidationError(f"Path is not a file: {pdf_path}")

        # Check file extension
        if pdf_path.suffix.lower() not in self.supported_formats:
            raise ValidationError(
                f"Unsupported file format: {pdf_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        # Check file MIME type
        mime_type, _ = mimetypes.guess_type(str(pdf_path))
        if mime_type and mime_type != 'application/pdf':
            raise ValidationError(f"File does not appear to be a PDF: {mime_type}")

        # Check file size
        file_size = pdf_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValidationError(
                f"File too large: {file_size:,} bytes (max: {self.max_file_size:,} bytes)"
            )

        if file_size == 0:
            raise ValidationError("File is empty")

        logger.debug(f"PDF file validation passed: {pdf_path}")
        return pdf_path

    def validate_output_path(self, output_path: Union[str, Path]) -> Path:
        """
        Validate output path and ensure parent directory exists.

        Args:
            output_path: Path where output should be written

        Returns:
            Validated Path object

        Raises:
            ValidationError: If output path is invalid
        """
        output_path = Path(output_path)

        # Check if parent directory exists or can be created
        parent_dir = output_path.parent
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory {parent_dir}: {e}")

        # Check if file is writable
        if output_path.exists():
            if not output_path.is_file():
                raise ValidationError(f"Output path exists but is not a file: {output_path}")
            if not os.access(output_path, os.W_OK):
                raise ValidationError(f"Output file is not writable: {output_path}")
        else:
            # Test write permissions by creating a test file
            test_file = output_path.with_suffix('.test')
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise ValidationError(f"Output directory is not writable: {e}")

        logger.debug(f"Output path validation passed: {output_path}")
        return output_path

    def validate_processing_parameters(
        self,
        backend_type: str,
        processing_mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate processing parameters.

        Args:
            backend_type: Backend type string
            processing_mode: Processing mode string
            **kwargs: Additional parameters

        Returns:
            Validated parameters dictionary

        Raises:
            ValidationError: If parameters are invalid
        """
        from ..backends.base import BackendType, ProcessingMode

        # Validate backend type
        try:
            validated_backend = BackendType(backend_type.lower())
        except ValueError:
            available_backends = [bt.value for bt in BackendType]
            raise ValidationError(
                f"Invalid backend type: {backend_type}. "
                f"Available backends: {', '.join(available_backends)}"
            )

        # Validate processing mode
        try:
            validated_mode = ProcessingMode(processing_mode.lower())
        except ValueError:
            available_modes = [pm.value for pm in ProcessingMode]
            raise ValidationError(
                f"Invalid processing mode: {processing_mode}. "
                f"Available modes: {', '.join(available_modes)}"
            )

        # Validate additional parameters
        validated_params = {
            'backend_type': validated_backend,
            'processing_mode': validated_mode
        }

        # Validate DPI if provided
        if 'dpi' in kwargs:
            dpi = kwargs['dpi']
            if not isinstance(dpi, int) or dpi < 72 or dpi > 600:
                raise ValidationError("DPI must be an integer between 72 and 600")
            validated_params['dpi'] = dpi

        # Validate parallel processing if provided
        if 'parallel_processing' in kwargs:
            parallel = kwargs['parallel_processing']
            if not isinstance(parallel, bool):
                raise ValidationError("parallel_processing must be a boolean")
            validated_params['parallel_processing'] = parallel

        # Validate max workers if provided
        if 'max_workers' in kwargs:
            max_workers = kwargs['max_workers']
            if not isinstance(max_workers, int) or max_workers < 1 or max_workers > 16:
                raise ValidationError("max_workers must be an integer between 1 and 16")
            validated_params['max_workers'] = max_workers

        logger.debug(f"Parameter validation passed: {validated_params}")
        return validated_params


class OutputValidator:
    """Validator for generated output and processing results."""

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the output validator.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self.min_content_length = config.processing.get('min_content_length', 10)
        self.max_content_length = config.processing.get('max_content_length', 100000)

    def validate_markdown_output(self, markdown: str) -> str:
        """
        Validate generated markdown content.

        Args:
            markdown: Generated markdown content

        Returns:
            Validated markdown content

        Raises:
            ValidationError: If markdown is invalid
        """
        if not markdown or not markdown.strip():
            raise ValidationError("Generated markdown is empty")

        # Check content length
        content_length = len(markdown.strip())
        if content_length < self.min_content_length:
            raise ValidationError(
                f"Generated content too short: {content_length} characters "
                f"(minimum: {self.min_content_length})"
            )

        if content_length > self.max_content_length:
            raise ValidationError(
                f"Generated content too long: {content_length} characters "
                f"(maximum: {self.max_content_length})"
            )

        # Basic markdown structure validation
        if not self._has_markdown_structure(markdown):
            logger.warning("Generated content may not be properly formatted as markdown")

        logger.debug(f"Markdown validation passed: {content_length} characters")
        return markdown

    def validate_document_structure(self, document: Document) -> Document:
        """
        Validate processed document structure.

        Args:
            document: Processed document object

        Returns:
            Validated document object

        Raises:
            ValidationError: If document structure is invalid
        """
        if not document.pages:
            raise ValidationError("Document has no pages")

        if document.page_count != len(document.pages):
            raise ValidationError(
                f"Page count mismatch: reported {document.page_count}, "
                f"actual {len(document.pages)}"
            )

        # Validate each page
        for i, page in enumerate(document.pages):
            self._validate_page(page, i)

        logger.debug(f"Document structure validation passed: {document.page_count} pages")
        return document

    def _validate_page(self, page: Page, page_index: int) -> None:
        """
        Validate a single page.

        Args:
            page: Page object
            page_index: Index of the page in document

        Raises:
            ValidationError: If page is invalid
        """
        if page.page_number != page_index:
            logger.warning(
                f"Page number mismatch: expected {page_index}, "
                f"found {page.page_number}"
            )

        if page.width <= 0 or page.height <= 0:
            raise ValidationError(
                f"Invalid page dimensions for page {page_index}: "
                f"{page.width}x{page.height}"
            )

        if page.dpi <= 0:
            raise ValidationError(
                f"Invalid DPI for page {page_index}: {page.dpi}"
            )

        # Validate content elements
        for element in page.elements:
            self._validate_content_element(element, page_index)

    def _validate_content_element(
        self,
        element: ContentElement,
        page_index: int
    ) -> None:
        """
        Validate a content element.

        Args:
            element: Content element object
            page_index: Index of the containing page

        Raises:
            ValidationError: If element is invalid
        """
        if not element.text or not element.text.strip():
            raise ValidationError(
                f"Empty content element on page {page_index}"
            )

        if not 0 <= element.confidence <= 1:
            raise ValidationError(
                f"Invalid confidence score on page {page_index}: "
                f"{element.confidence}"
            )

        # Validate bounding box if present
        if element.bounding_box:
            bbox = element.bounding_box
            if bbox.x0 >= bbox.x1 or bbox.y0 >= bbox.y1:
                raise ValidationError(
                    f"Invalid bounding box on page {page_index}: "
                    f"({bbox.x0}, {bbox.y0}, {bbox.x1}, {bbox.y1})"
                )

    def _has_markdown_structure(self, text: str) -> bool:
        """
        Check if text has basic markdown structure.

        Args:
            text: Text to check

        Returns:
            True if text appears to have markdown structure
        """
        # Check for common markdown patterns
        markdown_indicators = [
            '# ',  # Headers
            '## ', '### ', '#### ',
            '*', '**',  # Bold/italic
            '```',  # Code blocks
            '[', ']',  # Links
            '|',  # Tables
            '$',  # Math
            '\n- ',  # Lists
            '\n1. ',  # Numbered lists
        ]

        return any(indicator in text for indicator in markdown_indicators)


class IntegrityChecker:
    """Checker for data integrity and consistency."""

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the integrity checker.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config

    def check_processing_consistency(
        self,
        original_document: Document,
        processed_document: Document
    ) -> Dict[str, Any]:
        """
        Check consistency between original and processed documents.

        Args:
            original_document: Original document from PDF analysis
            processed_document: Document after processing

        Returns:
            Dictionary with consistency check results
        """
        results = {
            'page_count_match': True,
            'page_count_diff': 0,
            'content_preserved': True,
            'warnings': [],
            'errors': []
        }

        # Check page count
        if original_document.page_count != processed_document.page_count:
            results['page_count_match'] = False
            results['page_count_diff'] = abs(
                original_document.page_count - processed_document.page_count
            )
            results['errors'].append(
                f"Page count mismatch: {original_document.page_count} vs "
                f"{processed_document.page_count}"
            )

        # Check content preservation
        original_elements = original_document.get_total_elements()
        processed_elements = processed_document.get_total_elements()

        if processed_elements < original_elements * 0.5:  # Less than 50% preserved
            results['content_preserved'] = False
            results['warnings'].append(
                f"Significant content loss: {processed_elements}/{original_elements} elements preserved"
            )

        # Check for empty pages
        empty_pages = sum(1 for page in processed_document.pages if not page.elements)
        if empty_pages > 0:
            results['warnings'].append(f"{empty_pages} empty pages found")

        logger.debug(f"Integrity check completed: {results}")
        return results

    def check_backend_health(self, backend_name: str) -> Dict[str, Any]:
        """
        Check health and availability of a backend.

        Args:
            backend_name: Name of the backend to check

        Returns:
            Dictionary with health check results
        """
        results = {
            'backend': backend_name,
            'healthy': False,
            'response_time': None,
            'error': None,
            'available_models': 0
        }

        try:
            # This would be implemented with actual backend health checks
            # For now, return basic status
            results['healthy'] = True
            results['available_models'] = 1  # Placeholder

        except Exception as e:
            results['error'] = str(e)
            logger.warning(f"Backend health check failed for {backend_name}: {e}")

        return results
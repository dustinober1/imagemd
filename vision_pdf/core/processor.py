"""
Main VisionPDF processor for PDF to markdown conversion.

This module provides the primary interface for converting PDF documents
to markdown using vision language models.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import logging

from .document import Document, Page
from ..config.settings import VisionPDFConfig, BackendType, ProcessingMode
from ..backends.base import VLMBackend
from ..pdf.renderer import PDFRenderer
from ..pdf.analyzer import PDFAnalyzer
from ..pdf.extractor import PDFExtractor
from ..utils.logging_config import get_logger
from ..utils.exceptions import (
    VisionPDFError,
    PDFProcessingError,
    BackendError,
    ValidationError
)

logger = get_logger(__name__)


class VisionPDF:
    """
    Main class for PDF to markdown conversion using vision language models.

    This class orchestrates the entire conversion process, from PDF analysis
    and rendering to VLM processing and markdown generation.
    """

    def __init__(
        self,
        config: Optional[VisionPDFConfig] = None,
        backend_type: BackendType = BackendType.OLLAMA,
        backend_config: Optional[Dict[str, Any]] = None,
        processing_mode: ProcessingMode = ProcessingMode.HYBRID,
        **kwargs
    ):
        """
        Initialize VisionPDF processor.

        Args:
            config: VisionPDF configuration object
            backend_type: Type of VLM backend to use
            backend_config: Backend-specific configuration
            processing_mode: Processing mode for conversion
            **kwargs: Additional configuration options
        """
        # Initialize configuration
        if config is None:
            config = VisionPDFConfig()
            # Apply any additional configuration
            config.default_backend = backend_type
            config.processing.mode = processing_mode

            # Apply backend configuration if provided
            if backend_config:
                config.backends[backend_type.value].config.update(backend_config)

        self.config = config
        self.backend_type = backend_type
        self.processing_mode = processing_mode

        # Initialize components
        self._init_components()

        # VLM backend (lazy initialization)
        self._backend: Optional[VLMBackend] = None

        logger.info(f"VisionPDF initialized with backend: {backend_type.value}")

    def _init_components(self) -> None:
        """Initialize PDF processing components."""
        self.renderer = PDFRenderer(self.config)
        self.analyzer = PDFAnalyzer(self.config)
        self.extractor = PDFExtractor(self.config)

    async def _get_backend(self) -> VLMBackend:
        """Get or initialize the VLM backend."""
        if self._backend is None:
            await self._init_backend()
        return self._backend

    async def _init_backend(self) -> None:
        """Initialize the VLM backend."""
        try:
            # Import backend classes
            if self.backend_type == BackendType.OLLAMA:
                from ..backends.ollama import OllamaBackend
                backend_class = OllamaBackend
            elif self.backend_type == BackendType.LLAMA_CPP:
                from ..backends.llama_cpp import LlamaCppBackend
                backend_class = LlamaCppBackend
            elif self.backend_type == BackendType.CUSTOM_API:
                from ..backends.custom import CustomAPIBackend
                backend_class = CustomAPIBackend
            else:
                raise BackendError(f"Unsupported backend type: {self.backend_type}")

            # Get backend configuration
            backend_config = self.config.get_backend_config(self.backend_type)
            if backend_config is None:
                raise BackendError(f"No configuration found for backend: {self.backend_type}")

            # Initialize backend
            self._backend = backend_class(backend_config.config)
            await self._backend.initialize()

            logger.info(f"Initialized {self.backend_type.value} backend")

        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            raise BackendError(f"Backend initialization failed: {e}")

    async def convert_pdf(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Convert a PDF file to markdown string.

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback for progress updates

        Returns:
            Markdown content as string
        """
        pdf_path = Path(pdf_path)

        # Validate input
        self._validate_pdf_path(pdf_path)

        try:
            # Analyze document
            logger.info(f"Analyzing PDF: {pdf_path}")
            document = self.analyzer.analyze_document(pdf_path)

            # Process document
            markdown_content = await self._process_document(document, progress_callback)

            logger.info(f"Successfully converted {pdf_path}")
            return markdown_content

        except Exception as e:
            logger.error(f"Conversion failed for {pdf_path}: {e}")
            raise PDFProcessingError(f"Failed to convert PDF: {e}", str(pdf_path))

    async def convert_pdf_to_file(
        self,
        pdf_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Convert a PDF file to markdown and save to file.

        Args:
            pdf_path: Path to the input PDF file
            output_path: Path to the output markdown file
            progress_callback: Optional callback for progress updates
        """
        # Convert to markdown
        markdown_content = await self.convert_pdf(pdf_path, progress_callback)

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Markdown saved to: {output_path}")

    async def convert_batch(
        self,
        pdf_paths: List[str],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[str]:
        """
        Convert multiple PDF files to markdown.

        Args:
            pdf_paths: List of paths to PDF files
            output_dir: Directory to save output files
            progress_callback: Optional callback with (current, total, filename)

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(i, len(pdf_paths), Path(pdf_path).name)

            try:
                # Generate output filename
                pdf_name = Path(pdf_path).stem
                output_path = output_dir / f"{pdf_name}.md"

                # Convert PDF
                await self.convert_pdf_to_file(pdf_path, output_path)
                results.append(str(output_path))

            except Exception as e:
                logger.error(f"Failed to convert {pdf_path}: {e}")
                # Continue with other files

        if progress_callback:
            progress_callback(len(pdf_paths), len(pdf_paths), "Complete")

        return results

    async def _process_document(
        self,
        document: Document,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Process a document object to generate markdown.

        Args:
            document: Document object to process
            progress_callback: Optional progress callback

        Returns:
            Generated markdown content
        """
        backend = await self._get_backend()
        markdown_parts = []

        # Process each page
        for i, page in enumerate(document.pages):
            if progress_callback:
                progress_callback(i + 1, len(document.pages))

            page_markdown = await self._process_page(page, backend)
            markdown_parts.append(page_markdown)

        # Combine all pages
        return "\n\n".join(markdown_parts)

    async def _process_page(self, page: Page, backend: VLMBackend) -> str:
        """
        Process a single page using the VLM backend.

        Args:
            page: Page object to process
            backend: VLM backend instance

        Returns:
            Markdown content for the page
        """
        try:
            # Create processing request
            request = self._create_processing_request(page)

            # Process with VLM
            from ..backends.base import ProcessingRequest
            response = await backend.process_page(request)

            # Handle errors in response
            if response.error_message:
                logger.warning(f"VLM processing warning: {response.error_message}")
                # Fall back to text extraction
                return self._fallback_text_extraction(page)

            return response.markdown

        except Exception as e:
            logger.error(f"VLM processing failed for page {page.page_number}: {e}")
            # Fall back to text extraction
            return self._fallback_text_extraction(page)

    def _create_processing_request(self, page: Page) -> "ProcessingRequest":
        """Create a processing request for a page."""
        from ..backends.base import ProcessingRequest

        # Create temporary image file for the page
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir / f"page_{page.page_number}.png"

        try:
            # Render page to image
            # Note: This would require the original PDF path, which we don't have here
            # For now, we'll create a request without image data
            request = ProcessingRequest(
                image_path=image_path,
                text_content=page.raw_text,
                processing_mode=self.processing_mode,
                prompt=self._get_page_prompt(page)
            )

            return request

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_page_prompt(self, page: Page) -> str:
        """Get appropriate prompt for page processing."""
        base_prompt = (
            "Convert this PDF page to well-formatted markdown. "
            "Preserve the layout, tables, mathematical formulas, and code blocks. "
            "Use proper markdown formatting for headings, lists, and other elements."
        )

        if self.config.processing.preserve_tables:
            base_prompt += " Pay special attention to preserving table structures."

        if self.config.processing.preserve_math:
            base_prompt += " Convert mathematical expressions to LaTeX format."

        if self.config.processing.preserve_code:
            base_prompt += " Identify and format code blocks with appropriate syntax highlighting."

        return base_prompt

    def _fallback_text_extraction(self, page: Page) -> str:
        """
        Fall back to basic text extraction.

        Args:
            page: Page object

        Returns:
            Basic markdown from text extraction
        """
        if page.raw_text:
            return f"# Page {page.page_number + 1}\n\n{page.raw_text}"
        else:
            return f"# Page {page.page_number + 1}\n\n*Content extraction failed*"

    def _validate_pdf_path(self, pdf_path: Path) -> None:
        """Validate the PDF file path."""
        if not pdf_path.exists():
            raise ValidationError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise ValidationError(f"Path is not a file: {pdf_path}")

        if not str(pdf_path).lower().endswith('.pdf'):
            raise ValidationError(f"File is not a PDF: {pdf_path}")

        # Validate file size
        max_size = self.config.max_file_size
        file_size = pdf_path.stat().st_size
        if file_size > max_size:
            raise ValidationError(
                f"PDF file too large: {file_size} bytes (max: {max_size} bytes)"
            )

    async def close(self) -> None:
        """Clean up resources."""
        if self._backend:
            try:
                await self._backend.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up backend: {e}")
            finally:
                self._backend = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __enter__(self):
        """Sync context manager entry (for backward compatibility)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        # Note: In sync context, we can't properly clean up async resources
        logger.warning("Using sync context manager - consider using async context manager for proper cleanup")

    async def get_available_models(self) -> List[str]:
        """Get list of available models from the backend."""
        backend = await self._get_backend()
        models = await backend.get_available_models()
        return [model.name for model in models]

    async def test_backend_connection(self) -> bool:
        """Test connection to the VLM backend."""
        try:
            backend = await self._get_backend()
            return await backend.test_connection()
        except Exception as e:
            logger.error(f"Backend connection test failed: {e}")
            return False
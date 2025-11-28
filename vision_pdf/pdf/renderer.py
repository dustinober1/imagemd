"""
PDF rendering and image conversion utilities.

This module provides high-performance PDF to image conversion using PyMuPDF
with support for different DPI settings and image formats.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Optional, Tuple, Union
import io
from PIL import Image
import logging

from ..core.document import Page, BoundingBox
from ..config.settings import VisionPDFConfig

logger = logging.getLogger(__name__)


class PDFRenderer:
    """
    High-performance PDF renderer using PyMuPDF.

    This class handles converting PDF pages to high-quality images with
    configurable DPI, format, and rendering options.
    """

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize the PDF renderer.

        Args:
            config: VisionPDF configuration object
        """
        self.config = config
        self.dpi = config.processing.dpi
        self.temp_directory = Path(config.temp_directory)
        self.temp_directory.mkdir(parents=True, exist_ok=True)

    def render_page_to_image(
        self,
        pdf_document: fitz.Document,
        page_number: int,
        output_path: Optional[Path] = None,
        image_format: str = "PNG",
        **kwargs
    ) -> Path:
        """
        Render a single PDF page to an image file.

        Args:
            pdf_document: PyMuPDF document object
            page_number: Page number to render (0-based)
            output_path: Output path for the image file
            image_format: Image format (PNG, JPEG, etc.)
            **kwargs: Additional rendering options

        Returns:
            Path to the rendered image file
        """
        if page_number < 0 or page_number >= len(pdf_document):
            raise ValueError(f"Invalid page number: {page_number}")

        page = pdf_document[page_number]

        # Generate output path if not provided
        if output_path is None:
            output_path = self._generate_temp_path(page_number, image_format.lower())

        # Set up rendering parameters
        zoom = self.dpi / 72.0  # PDF default DPI is 72
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
            clip=kwargs.get('clip')
        )

        # Save pixmap to file
        if image_format.upper() == "PNG":
            pix.save(output_path)
        elif image_format.upper() == "JPEG":
            pix.save(output_path, jpeg_quality=kwargs.get('quality', 90))
        else:
            pix.save(output_path)

        logger.debug(f"Rendered page {page_number} to {output_path} (DPI: {self.dpi})")
        return output_path

    def render_page_to_pil_image(
        self,
        pdf_document: fitz.Document,
        page_number: int,
        **kwargs
    ) -> Image.Image:
        """
        Render a single PDF page to a PIL Image object.

        Args:
            pdf_document: PyMuPDF document object
            page_number: Page number to render (0-based)
            **kwargs: Additional rendering options

        Returns:
            PIL Image object
        """
        if page_number < 0 or page_number >= len(pdf_document):
            raise ValueError(f"Invalid page number: {page_number}")

        page = pdf_document[page_number]

        # Set up rendering parameters
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert pixmap to PIL Image
        img_data = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))

        logger.debug(f"Rendered page {page_number} to PIL Image (size: {img.size})")
        return img

    def render_document_to_images(
        self,
        pdf_path: Union[str, Path],
        output_directory: Optional[Path] = None,
        image_format: str = "PNG",
        page_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> List[Path]:
        """
        Render multiple pages of a PDF document to image files.

        Args:
            pdf_path: Path to the PDF file
            output_directory: Directory to save images (uses temp directory if None)
            image_format: Image format for output files
            page_range: Tuple of (start_page, end_page) to render specific range
            **kwargs: Additional rendering options

        Returns:
            List of paths to rendered image files
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Set output directory
        if output_directory is None:
            output_directory = self.temp_directory / "images"
        output_directory.mkdir(parents=True, exist_ok=True)

        # Open PDF document
        pdf_document = fitz.open(str(pdf_path))
        try:
            # Determine page range
            total_pages = len(pdf_document)
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(total_pages, end_page + 1)  # +1 for inclusive range
                pages_to_render = range(start_page, end_page)
            else:
                pages_to_render = range(total_pages)

            # Render each page
            image_paths = []
            for page_num in pages_to_render:
                output_path = output_directory / f"page_{page_num:04d}.{image_format.lower()}"
                rendered_path = self.render_page_to_image(
                    pdf_document=pdf_document,
                    page_number=page_num,
                    output_path=output_path,
                    image_format=image_format,
                    **kwargs
                )
                image_paths.append(rendered_path)

            logger.info(f"Rendered {len(image_paths)} pages from {pdf_path}")
            return image_paths

        finally:
            pdf_document.close()

    def get_page_dimensions(
        self,
        pdf_path: Union[str, Path],
        page_number: int
    ) -> Tuple[float, float]:
        """
        Get the dimensions of a PDF page in points.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-based)

        Returns:
            Tuple of (width, height) in points
        """
        pdf_path = Path(pdf_path)
        pdf_document = fitz.open(str(pdf_path))
        try:
            if page_number < 0 or page_number >= len(pdf_document):
                raise ValueError(f"Invalid page number: {page_number}")

            page = pdf_document[page_number]
            rect = page.rect
            return rect.width, rect.height
        finally:
            pdf_document.close()

    def estimate_rendering_time(
        self,
        pdf_path: Union[str, Path],
        page_range: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Estimate the time required to render a PDF document.

        Args:
            pdf_path: Path to the PDF file
            page_range: Page range to render (None for all pages)

        Returns:
            Estimated rendering time in seconds
        """
        pdf_path = Path(pdf_path)
        pdf_document = fitz.open(str(pdf_path))
        try:
            total_pages = len(pdf_document)
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(total_pages, end_page + 1)
                pages_to_render = end_page - start_page
            else:
                pages_to_render = total_pages

            # Estimate based on DPI and page count
            # Rough estimate: 0.5 seconds per page at 300 DPI, scaling with DPI
            base_time_per_page = 0.5 * (self.dpi / 300)
            estimated_time = pages_to_render * base_time_per_page

            logger.debug(f"Estimated rendering time for {pages_to_render} pages: {estimated_time:.2f}s")
            return estimated_time
        finally:
            pdf_document.close()

    def _generate_temp_path(self, page_number: int, image_format: str) -> Path:
        """
        Generate a temporary file path for a rendered image.

        Args:
            page_number: Page number
            image_format: Image format

        Returns:
            Path to temporary file
        """
        return self.temp_directory / f"render_page_{page_number:04d}.{image_format}"

    def cleanup_temp_files(self) -> None:
        """Clean up temporary image files."""
        try:
            import glob
            temp_files = glob.glob(str(self.temp_directory / "render_page_*"))
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)
            logger.debug(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    def validate_pdf(self, pdf_path: Union[str, Path]) -> bool:
        """
        Validate that a file is a readable PDF.

        Args:
            pdf_path: Path to the file to validate

        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return False

            # Try to open with PyMuPDF
            pdf_document = fitz.open(str(pdf_path))
            page_count = len(pdf_document)
            pdf_document.close()

            # Check if it has pages
            return page_count > 0

        except Exception as e:
            logger.warning(f"PDF validation failed for {pdf_path}: {e}")
            return False
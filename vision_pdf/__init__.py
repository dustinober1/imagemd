"""
VisionPDF: Convert PDF documents to well-formatted markdown using vision language models.

This package provides a flexible, extensible framework for PDF to markdown conversion
with support for multiple vision language model backends including Ollama, llama.cpp,
and custom API endpoints.

Key Features:
- User-configurable vision language models
- Multiple backend support (Ollama, llama.cpp, custom APIs)
- Processing flexibility (vision-only, hybrid, text-only modes)
- Full format preservation (tables, math formulas, code blocks, layouts)
- Enterprise ready with internal system support
- High performance with parallel processing and caching

Basic Usage:
    from vision_pdf import VisionPDF

    converter = VisionPDF()
    markdown = converter.convert_pdf("document.pdf")
    converter.convert_pdf_to_file("document.pdf", "output.md")

Advanced Usage:
    from vision_pdf import VisionPDF, BackendType, ProcessingMode

    converter = VisionPDF(
        backend_type=BackendType.OLLAMA,
        processing_mode=ProcessingMode.HYBRID,
        parallel_processing=True,
        preserve_tables=True,
        preserve_math=True
    )

    markdown = converter.convert_pdf("technical_doc.pdf")
"""

__version__ = "1.0.0"
__author__ = "VisionPDF Team"
__email__ = "contact@visionpdf.com"
__description__ = "Convert PDF documents to well-formatted markdown using vision language models"

# Main exports
from .core.processor import VisionPDF
from .core.document import Document, Page
from .backends.base import BackendType, ProcessingMode
from .config.settings import VisionPDFConfig

__all__ = [
    "VisionPDF",
    "Document",
    "Page",
    "BackendType",
    "ProcessingMode",
    "VisionPDFConfig",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]
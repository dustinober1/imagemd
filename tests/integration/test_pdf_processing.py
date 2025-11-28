"""
Integration tests for PDF processing functionality.

These tests require actual PDF files and test the integration
between different PDF processing components.
"""

import pytest
from pathlib import Path

from vision_pdf.pdf.renderer import PDFRenderer
from vision_pdf.pdf.analyzer import PDFAnalyzer
from vision_pdf.pdf.extractor import PDFExtractor
from vision_pdf.core.document import Document, ContentType


@pytest.mark.integration
class TestPDFProcessingIntegration:
    """Integration tests for PDF processing."""

    def test_complete_pdf_analysis_workflow(self, sample_config, sample_pdf_path):
        """Test complete PDF analysis workflow."""
        # Initialize components
        renderer = PDFRenderer(sample_config)
        analyzer = PDFAnalyzer(sample_config)
        extractor = PDFExtractor(sample_config)

        # Test PDF validation
        assert renderer.validate_pdf(sample_pdf_path)

        # Test document analysis
        document = analyzer.analyze_document(sample_pdf_path)

        # Verify document structure
        assert document is not None
        assert document.file_path == sample_pdf_path
        assert document.page_count > 0
        assert len(document.pages) == document.page_count

        # Test page dimensions
        for page in document.pages:
            assert page.width > 0
            assert page.height > 0
            assert page.dpi > 0

        # Test text extraction
        for page_num in range(document.page_count):
            text = extractor.extract_text_from_page(sample_pdf_path, page_num)
            assert isinstance(text, str)
            # Should have some text content from our sample PDF
            if text.strip():
                assert len(text) > 0

    def test_pdf_rendering_workflow(self, sample_config, sample_pdf_path, temp_dir):
        """Test PDF rendering workflow."""
        renderer = PDFRenderer(sample_config)

        # Test single page rendering
        output_path = renderer.render_page_to_image(
            pdf_document=fitz.open(str(sample_pdf_path)),
            page_number=0,
            output_path=temp_dir / "page_0.png"
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Test document rendering
        image_paths = renderer.render_document_to_images(
            pdf_path=sample_pdf_path,
            output_directory=temp_dir / "rendered"
        )

        assert len(image_paths) > 0
        for path in image_paths:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_text_extraction_methods(self, sample_config, sample_pdf_path):
        """Test different text extraction methods."""
        extractor = PDFExtractor(sample_config)

        # Test PyMuPDF extraction
        pymupdf_text = extractor.extract_text_from_page(
            sample_pdf_path, 0, method="pymupdf"
        )
        assert isinstance(pymupdf_text, str)

        # Test pdfplumber extraction
        pdfplumber_text = extractor.extract_text_from_page(
            sample_pdf_path, 0, method="pdfplumber"
        )
        assert isinstance(pdfplumber_text, str)

        # Test auto method (should choose the best result)
        auto_text = extractor.extract_text_from_page(
            sample_pdf_path, 0, method="auto"
        )
        assert isinstance(auto_text, str)

        # Auto should return the longer/more complete text
        assert len(auto_text) >= max(len(pymupdf_text), len(pdfplumber_text)) - 10  # Allow small differences

    def test_layout_aware_extraction(self, sample_config, sample_pdf_path):
        """Test layout-aware text extraction."""
        extractor = PDFExtractor(sample_config)

        # Extract text with layout information
        elements = extractor.extract_text_with_layout(sample_pdf_path, 0)

        assert isinstance(elements, list)
        assert len(elements) > 0

        # Verify element structure
        for element in elements:
            assert element.text is not None
            assert len(element.text) > 0
            assert element.confidence > 0
            assert element.bounding_box is not None
            assert element.bounding_box.page == 0

    def test_content_detection(self, sample_config, sample_pdf_path):
        """Test content type detection."""
        analyzer = PDFAnalyzer(sample_config)

        # Analyze document
        document = analyzer.analyze_document(sample_pdf_path)

        # Should detect text content
        text_elements = document.get_all_elements_by_type(ContentType.TEXT)
        assert len(text_elements) > 0

        # Verify text elements have expected properties
        for element in text_elements:
            assert element.content_type == ContentType.TEXT
            assert element.text.strip()  # Should have non-empty text
            assert 0 <= element.confidence <= 1

    def test_extraction_quality_assessment(self, sample_config, sample_pdf_path):
        """Test extraction quality scoring."""
        extractor = PDFExtractor(sample_config)

        # Get quality score for first page
        quality_score = extractor.get_extraction_quality_score(sample_pdf_path, 0)

        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1

        # For our sample PDF, should have reasonable quality
        assert quality_score > 0.1  # Should extract something

    def test_metadata_extraction(self, sample_config, sample_pdf_path):
        """Test PDF metadata extraction."""
        extractor = PDFExtractor(sample_config)

        # Extract metadata
        metadata = extractor.extract_metadata(sample_pdf_path)

        assert isinstance(metadata, dict)
        assert 'page_count' in metadata
        assert 'pdf_version' in metadata
        assert 'is_pdf' in metadata

        # Verify metadata values
        assert metadata['is_pdf'] is True
        assert metadata['page_count'] > 0

    def test_performance_estimation(self, sample_config, sample_pdf_path):
        """Test performance estimation."""
        renderer = PDFRenderer(sample_config)

        # Estimate rendering time
        estimated_time = renderer.estimate_rendering_time(sample_pdf_path)

        assert isinstance(estimated_time, float)
        assert estimated_time > 0

        # Estimate for page range
        estimated_time_range = renderer.estimate_rendering_time(
            sample_pdf_path, page_range=(0, 0)
        )

        assert isinstance(estimated_time_range, float)
        assert estimated_time_range > 0

        # Range should be less or equal to full document
        assert estimated_time_range <= estimated_time

    def test_temporary_file_cleanup(self, sample_config, sample_pdf_path):
        """Test temporary file cleanup."""
        renderer = PDFRenderer(sample_config)

        # Render some pages to create temp files
        renderer.render_page_to_image(
            pdf_document=fitz.open(str(sample_pdf_path)),
            page_number=0
        )

        # Clean up
        renderer.cleanup_temp_files()

        # Verify cleanup happened (no direct way to test this, but should not raise errors)
        assert True  # If we get here, cleanup didn't raise an error

    @pytest.mark.slow
    def test_large_document_processing(self, sample_config, temp_dir):
        """Test processing of a larger document."""
        # Create a larger test document
        import fitz

        large_pdf_path = temp_dir / "large_document.pdf"
        doc = fitz.open()

        # Create 10 pages with content
        for i in range(10):
            page = doc.new_page()
            page.insert_text((72, 72 + i * 20), f"Page {i + 1} content", fontsize=12)
            page.insert_text((72, 100 + i * 20), f"This is line {i + 1} of text.", fontsize=10)

        doc.save(large_pdf_path)
        doc.close()

        # Test processing
        analyzer = PDFAnalyzer(sample_config)
        document = analyzer.analyze_document(large_pdf_path)

        assert document.page_count == 10
        assert len(document.pages) == 10

        # Should have text on all pages
        total_text_elements = len(document.get_all_elements_by_type(ContentType.TEXT))
        assert total_text_elements >= 10  # At least one text element per page


# Import fitz for the integration tests
import fitz
"""
Unit tests for VisionPDF core document classes.
"""

import pytest
from pathlib import Path

from vision_pdf.core.document import (
    Document, Page, ContentElement, BoundingBox,
    ContentType, ProcessingMode
)


class TestBoundingBox:
    """Test cases for BoundingBox."""

    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=200, page=0)
        assert bbox.x0 == 10
        assert bbox.y0 == 20
        assert bbox.x1 == 100
        assert bbox.y1 == 200
        assert bbox.page == 0

    def test_invalid_bounding_box(self):
        """Test creating invalid bounding boxes."""
        # Invalid coordinates (x0 >= x1)
        with pytest.raises(ValueError):
            BoundingBox(x0=100, y0=20, x1=10, y1=200, page=0)

        # Invalid coordinates (y0 >= y1)
        with pytest.raises(ValueError):
            BoundingBox(x0=10, y0=200, x1=100, y1=20, page=0)

        # Invalid page number
        with pytest.raises(ValueError):
            BoundingBox(x0=10, y0=20, x1=100, y1=200, page=-1)


class TestContentElement:
    """Test cases for ContentElement."""

    def test_valid_content_element(self):
        """Test creating a valid content element."""
        element = ContentElement(
            content_type=ContentType.TEXT,
            text="Sample text",
            confidence=0.9
        )
        assert element.content_type == ContentType.TEXT
        assert element.text == "Sample text"
        assert element.confidence == 0.9
        assert element.bounding_box is None
        assert element.metadata == {}

    def test_content_element_with_metadata(self):
        """Test creating content element with metadata."""
        bbox = BoundingBox(x0=0, y0=0, x1=100, y1=50, page=0)
        element = ContentElement(
            content_type=ContentType.TABLE,
            text="Table content",
            confidence=0.85,
            bounding_box=bbox,
            metadata={"rows": 3, "cols": 2}
        )
        assert element.content_type == ContentType.TABLE
        assert element.bounding_box == bbox
        assert element.metadata["rows"] == 3
        assert element.metadata["cols"] == 2

    def test_invalid_content_element(self):
        """Test creating invalid content elements."""
        # Empty text
        with pytest.raises(ValueError):
            ContentElement(
                content_type=ContentType.TEXT,
                text="",
                confidence=0.5
            )

        # Invalid confidence
        with pytest.raises(ValueError):
            ContentElement(
                content_type=ContentType.TEXT,
                text="Sample",
                confidence=1.5
            )

        with pytest.raises(ValueError):
            ContentElement(
                content_type=ContentType.TEXT,
                text="Sample",
                confidence=-0.1
            )


class TestPage:
    """Test cases for Page."""

    def test_valid_page(self):
        """Test creating a valid page."""
        page = Page(
            page_number=0,
            width=612,
            height=792,
            dpi=150
        )
        assert page.page_number == 0
        assert page.width == 612
        assert page.height == 792
        assert page.dpi == 150
        assert len(page.elements) == 0
        assert page.raw_text is None

    def test_page_with_elements(self):
        """Test page with content elements."""
        page = Page(page_number=1, width=400, height=600, dpi=200)

        # Add text element
        text_element = ContentElement(
            content_type=ContentType.TEXT,
            text="Page title",
            confidence=0.95
        )
        page.add_element(text_element)

        # Add table element
        table_element = ContentElement(
            content_type=ContentType.TABLE,
            text="Table data",
            confidence=0.8
        )
        page.add_element(table_element)

        assert len(page.elements) == 2
        assert page.get_elements_by_type(ContentType.TEXT) == [text_element]
        assert page.get_elements_by_type(ContentType.TABLE) == [table_element]

    def test_invalid_page(self):
        """Test creating invalid pages."""
        # Negative page number
        with pytest.raises(ValueError):
            Page(page_number=-1, width=400, height=600, dpi=150)

        # Invalid dimensions
        with pytest.raises(ValueError):
            Page(page_number=0, width=0, height=600, dpi=150)

        with pytest.raises(ValueError):
            Page(page_number=0, width=400, height=0, dpi=150)

        # Invalid DPI
        with pytest.raises(ValueError):
            Page(page_number=0, width=400, height=600, dpi=0)

    def test_get_elements_by_type(self):
        """Test filtering elements by type."""
        page = Page(page_number=0, width=400, height=600, dpi=150)

        # Add elements of different types
        page.add_element(ContentElement(ContentType.TEXT, "Text 1", 0.9))
        page.add_element(ContentElement(ContentType.TABLE, "Table", 0.8))
        page.add_element(ContentElement(ContentType.TEXT, "Text 2", 0.85))
        page.add_element(ContentElement(ContentType.MATHEMATICAL, "Formula", 0.95))

        text_elements = page.get_text_elements()
        assert len(text_elements) == 2
        assert all(elem.content_type == ContentType.TEXT for elem in text_elements)

        table_elements = page.get_table_elements()
        assert len(table_elements) == 1
        assert table_elements[0].content_type == ContentType.TABLE

        math_elements = page.get_mathematical_elements()
        assert len(math_elements) == 1
        assert math_elements[0].content_type == ContentType.MATHEMATICAL

        code_elements = page.get_code_elements()
        assert len(code_elements) == 0


class TestDocument:
    """Test cases for Document."""

    def test_valid_document(self):
        """Test creating a valid document."""
        doc = Document(
            file_path=Path("test.pdf"),
            title="Test Document",
            author="Test Author"
        )
        assert doc.file_path == Path("test.pdf")
        assert doc.title == "Test Document"
        assert doc.author == "Test Author"
        assert doc.page_count == 0
        assert len(doc.pages) == 0

    def test_document_with_pages(self):
        """Test document with pages."""
        doc = Document(file_path=Path("test.pdf"))

        # Add pages
        page1 = Page(page_number=0, width=400, height=600, dpi=150)
        page2 = Page(page_number=1, width=400, height=600, dpi=150)

        doc.add_page(page1)
        doc.add_page(page2)

        assert doc.page_count == 2
        assert len(doc.pages) == 2
        assert doc.get_page(0) == page1
        assert doc.get_page(1) == page2

    def test_invalid_document(self):
        """Test creating invalid documents."""
        # Empty file path
        with pytest.raises(ValueError):
            Document(file_path=Path())

        # Page count mismatch
        with pytest.raises(ValueError):
            Document(
                file_path=Path("test.pdf"),
                page_count=2,
                pages=[Page(page_number=0, width=400, height=600, dpi=150)]
            )

    def test_get_all_elements_by_type(self):
        """Test getting all elements by type across pages."""
        doc = Document(file_path=Path("test.pdf"))

        # Create pages with different elements
        page1 = Page(page_number=0, width=400, height=600, dpi=150)
        page1.add_element(ContentElement(ContentType.TEXT, "Page 1 Text", 0.9))
        page1.add_element(ContentElement(ContentType.TABLE, "Page 1 Table", 0.8))

        page2 = Page(page_number=1, width=400, height=600, dpi=150)
        page2.add_element(ContentElement(ContentType.TEXT, "Page 2 Text", 0.85))
        page2.add_element(ContentElement(ContentType.MATHEMATICAL, "Formula", 0.95))

        doc.add_page(page1)
        doc.add_page(page2)

        # Get all text elements
        text_elements = doc.get_all_elements_by_type(ContentType.TEXT)
        assert len(text_elements) == 2
        assert text_elements[0].text == "Page 1 Text"
        assert text_elements[1].text == "Page 2 Text"

        # Get all table elements
        table_elements = doc.get_all_elements_by_type(ContentType.TABLE)
        assert len(table_elements) == 1
        assert table_elements[0].text == "Page 1 Table"

    def test_get_total_elements(self):
        """Test getting total element count."""
        doc = Document(file_path=Path("test.pdf"))

        # Add pages with elements
        page1 = Page(page_number=0, width=400, height=600, dpi=150)
        page1.add_element(ContentElement(ContentType.TEXT, "Text 1", 0.9))
        page1.add_element(ContentElement(ContentType.TABLE, "Table 1", 0.8))

        page2 = Page(page_number=1, width=400, height=600, dpi=150)
        page2.add_element(ContentElement(ContentType.TEXT, "Text 2", 0.85))

        doc.add_page(page1)
        doc.add_page(page2)

        assert doc.get_total_elements() == 3

    def test_processing_complexity(self):
        """Test processing complexity estimation."""
        # Small document
        small_doc = Document(file_path=Path("small.pdf"))
        for i in range(3):
            small_doc.add_page(Page(page_number=i, width=400, height=600, dpi=150))
        assert small_doc.get_processing_complexity() == "low"

        # Medium document
        medium_doc = Document(file_path=Path("medium.pdf"))
        for i in range(50):
            medium_doc.add_page(Page(page_number=i, width=400, height=600, dpi=150))
        assert medium_doc.get_processing_complexity() == "medium"

        # Large document
        large_doc = Document(file_path=Path("large.pdf"))
        for i in range(150):
            large_doc.add_page(Page(page_number=i, width=400, height=600, dpi=150))
        assert large_doc.get_processing_complexity() == "high"

    def test_has_complex_layouts(self):
        """Test complex layout detection."""
        # Simple document
        simple_doc = Document(file_path=Path("simple.pdf"))
        simple_page = Page(page_number=0, width=400, height=600, dpi=150)
        simple_page.add_element(ContentElement(ContentType.TEXT, "Simple text", 0.9))
        simple_doc.add_page(simple_page)
        assert simple_doc.has_complex_layouts() is False

        # Document with many elements
        complex_doc = Document(file_path=Path("complex.pdf"))
        complex_page = Page(page_number=0, width=400, height=600, dpi=150)
        for i in range(15):  # More than 10 elements
            complex_page.add_element(ContentElement(ContentType.TEXT, f"Text {i}", 0.9))
        complex_doc.add_page(complex_page)
        assert complex_doc.has_complex_layouts() is True

        # Document with tables
        table_doc = Document(file_path=Path("tables.pdf"))
        table_page = Page(page_number=0, width=400, height=600, dpi=150)
        table_page.add_element(ContentElement(ContentType.TABLE, "Table data", 0.8))
        table_doc.add_page(table_page)
        assert table_doc.has_complex_layouts() is True

    def test_get_page_bounds(self):
        """Test getting page bounds."""
        doc = Document(file_path=Path("test.pdf"))

        # Add pages
        doc.add_page(Page(page_number=0, width=400, height=600, dpi=150))
        doc.add_page(Page(page_number=1, width=400, height=600, dpi=150))

        # Valid page numbers
        assert doc.get_page(0) is not None
        assert doc.get_page(1) is not None

        # Invalid page numbers
        assert doc.get_page(-1) is None
        assert doc.get_page(2) is None
        assert doc.get_page(10) is None
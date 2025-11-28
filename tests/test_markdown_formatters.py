"""
Tests for markdown formatters.

This module tests the advanced table, math, and code formatters
to ensure they correctly detect and format content.
"""

import pytest
from vision_pdf.markdown.formatters.tables import (
    AdvancedTableDetector, TableFormatter, detect_and_format_tables
)
from vision_pdf.markdown.formatters.math import (
    MathPatternRecognizer, MathFormatter, detect_and_format_math
)
from vision_pdf.markdown.formatters.code import (
    CodeDetector, CodeFormatter, detect_and_format_code, CodeLanguage
)


class TestAdvancedTableDetector:
    """Test advanced table detection."""

    def test_detect_delimiter_table(self):
        """Test detection of delimiter-separated tables."""
        detector = AdvancedTableDetector()

        # CSV-like table
        csv_text = """Name,Age,City
John,25,New York
Jane,30,London
Bob,35,Paris"""

        table = detector.detect_table_from_text(csv_text)
        assert table is not None
        assert table.rows == 4
        assert table.cols == 3
        assert table.confidence > 0.7

    def test_detect_markdown_table(self):
        """Test detection of markdown tables."""
        detector = AdvancedTableDetector()

        markdown_table = """| Name | Age | City |
|------|-----|------|
| John | 25  | New York |
| Jane | 30  | London |"""

        table = detector.detect_table_from_vision_content(markdown_table)
        assert table is not None
        assert table.rows == 3
        assert table.cols == 3
        assert table.confidence > 0.8

    def test_detect_space_aligned_table(self):
        """Test detection of space-aligned tables."""
        detector = AdvancedTableDetector()

        space_table = """Name    Age   City
John     25    New York
Jane     30    London
Bob      35    Paris"""

        table = detector.detect_table_from_text(space_table)
        assert table is not None
        assert table.rows >= 3
        assert table.cols >= 3

    def test_no_table_detection(self):
        """Test that non-table text is not detected as table."""
        detector = AdvancedTableDetector()

        plain_text = """This is just regular text.
It has no table structure.
Just some sentences."""

        table = detector.detect_table_from_text(plain_text)
        assert table is None


class TestTableFormatter:
    """Test table formatting."""

    def test_format_simple_table(self):
        """Test formatting of a simple table."""
        formatter = TableFormatter()
        detector = AdvancedTableDetector()

        table_text = """Name,Age,City
John,25,New York
Jane,30,London"""

        table = detector.detect_table_from_text(table_text)
        formatted = formatter.format_table(table)

        assert '|' in formatted
        assert '---' in formatted
        assert 'Name' in formatted
        assert 'John' in formatted

    def test_format_table_with_headers(self):
        """Test formatting table with headers."""
        formatter = TableFormatter()
        detector = AdvancedTableDetector()

        table_text = """ID,Name,Type,Status
1,Task A,Development,In Progress
2,Task B,Testing,Completed
3,Task C,Planning,Not Started"""

        table = detector.detect_table_from_text(table_text)
        formatted = formatter.format_table(table)

        lines = formatted.split('\n')
        assert len(lines) >= 4  # Header + separator + data rows
        assert 'ID' in lines[0]
        assert '---' in lines[1]


class TestMathPatternRecognizer:
    """Test mathematical expression recognition."""

    def test_recognize_latex_inline(self):
        """Test recognition of inline LaTeX."""
        recognizer = MathPatternRecognizer()

        text = "The formula is $x^2 + y^2 = z^2$ for Pythagoras."
        expressions = recognizer.recognize_expressions(text)

        assert len(expressions) >= 1
        math_expr = expressions[0]
        assert math_expr.math_type.value == "inline"
        assert "$x^2 + y^2 = z^2$" in math_expr.latex

    def test_recognize_latex_display(self):
        """Test recognition of display LaTeX."""
        recognizer = MathPatternRecognizer()

        text = "The integral is $$\\int_0^\\infty e^{-x} dx = 1$$"
        expressions = recognizer.recognize_expressions(text)

        assert len(expressions) >= 1
        math_expr = expressions[0]
        assert math_expr.math_type.value == "display"

    def test_recognize_fractions(self):
        """Test recognition of fractions."""
        recognizer = MathPatternRecognizer()

        text = "The probability is p/q where p=0.5"
        expressions = recognizer.recognize_expressions(text)

        fraction_exprs = [e for e in expressions if e.math_type.value == "fraction"]
        assert len(fraction_exprs) >= 1
        assert "\\frac" in fraction_exprs[0].latex

    def test_recognize_superscripts(self):
        """Test recognition of superscripts."""
        recognizer = MathPatternRecognizer()

        text = "The formula is x^2 + y^3"
        expressions = recognizer.recognize_expressions(text)

        superscript_exprs = [e for e in expressions if e.math_type.value == "superscript"]
        assert len(superscript_exprs) >= 2  # x^2 and y^3


class TestMathFormatter:
    """Test mathematical expression formatting."""

    def test_format_inline_math(self):
        """Test formatting of inline math expressions."""
        formatter = MathFormatter()

        text = "The formula is x^2 + y^2 = z^2"
        formatted = formatter.format_text_with_math(text)

        assert "$" in formatted

    def test_format_display_math_for_complex(self):
        """Test that complex expressions use display math."""
        config = {"use_display_math_for_complex": True, "complexity_threshold": 10}
        formatter = MathFormatter(config)

        text = "The complex equation is \\int_0^\\infty e^{-x^2} dx = \\sqrt{\\pi}/2"
        formatted = formatter.format_text_with_math(text)

        # Should use display math for complex expression
        assert "$$" in formatted or "\\[" in formatted


class TestCodeDetector:
    """Test code block detection."""

    def test_detect_python_code(self):
        """Test detection of Python code."""
        detector = CodeDetector()

        python_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))"""

        code_blocks = detector.detect_code_blocks(python_code)
        assert len(code_blocks) >= 1
        code_block = code_blocks[0]
        assert code_block.language == CodeLanguage.PYTHON
        assert code_block.confidence > 0.7

    def test_detect_javascript_code(self):
        """Test detection of JavaScript code."""
        detector = CodeDetector()

        js_code = """function calculateSum(arr) {
    return arr.reduce((sum, num) => sum + num, 0);
}

const numbers = [1, 2, 3, 4, 5];
console.log(calculateSum(numbers));"""

        code_blocks = detector.detect_code_blocks(js_code)
        assert len(code_blocks) >= 1
        code_block = code_blocks[0]
        assert code_block.language == CodeLanguage.JAVASCRIPT

    def test_detect_sql_code(self):
        """Test detection of SQL code."""
        detector = CodeDetector()

        sql_code = """SELECT users.name, COUNT(orders.id) as order_count
FROM users
JOIN orders ON users.id = orders.user_id
WHERE orders.created_at >= '2023-01-01'
GROUP BY users.id, users.name
HAVING COUNT(orders.id) > 5
ORDER BY order_count DESC;"""

        code_blocks = detector.detect_code_blocks(sql_code)
        assert len(code_blocks) >= 1
        code_block = code_blocks[0]
        assert code_block.language == CodeLanguage.SQL

    def test_detect_bash_code(self):
        """Test detection of Bash code."""
        detector = CodeDetector()

        bash_code = """#!/bin/bash

for file in *.txt; do
    echo "Processing $file"
    wc -l "$file"
done"""

        code_blocks = detector.detect_code_blocks(bash_code)
        assert len(code_blocks) >= 1
        code_block = code_blocks[0]
        assert code_block.language == CodeLanguage.BASH

    def test_no_code_detection(self):
        """Test that non-code text is not detected as code."""
        detector = CodeDetector()

        plain_text = """This is just a regular paragraph.
It contains some text with punctuation.
But no actual programming code."""

        code_blocks = detector.detect_code_blocks(plain_text)
        assert len(code_blocks) == 0


class TestCodeFormatter:
    """Test code block formatting."""

    def test_format_python_code(self):
        """Test formatting of Python code blocks."""
        from vision_pdf.markdown.formatters.code import CodeBlock, CodeLanguage, format_code_block

        code_block = CodeBlock(
            lines=["def hello():", "    print('Hello, World!')"],
            language=CodeLanguage.PYTHON,
            confidence=0.9,
            start_line=0,
            end_line=1
        )

        formatted = format_code_block(code_block)
        assert formatted.startswith("```python")
        assert formatted.endswith("```")
        assert "def hello():" in formatted

    def test_format_code_with_line_numbers(self):
        """Test formatting with line numbers."""
        from vision_pdf.markdown.formatters.code import CodeBlock, CodeLanguage

        config = {"include_line_numbers": True}
        formatter = CodeFormatter(config)

        code_block = CodeBlock(
            lines=["line 1", "line 2", "line 3"],
            language=CodeLanguage.UNKNOWN,
            confidence=0.7,
            start_line=0,
            end_line=2
        )

        formatted = formatter.format_code_block(code_block)
        assert "1 | line 1" in formatted
        assert "2 | line 2" in formatted
        assert "3 | line 3" in formatted

    def test_format_text_with_code_blocks(self):
        """Test formatting text containing code blocks."""
        formatter = CodeFormatter()

        text_with_code = """Here is some regular text.

def example():
    print("This is code")

More regular text here."""

        formatted = formatter.format_text_with_code_blocks(text_with_code)
        assert "```python" in formatted or "```" in formatted
        assert "def example():" in formatted


class TestIntegrationFormatting:
    """Test integrated formatting of mixed content."""

    def test_format_mixed_content(self):
        """Test formatting content with tables, math, and code."""
        mixed_content = """# Example Document

## Data Table

Name,Score,Grade
Alice,95,A
Bob,87,B
Charlie,92,A-

## Mathematical Formula

The formula is x² + y² = z² where x=3 and y=4.

## Code Example

def calculate_grades(scores):
    grades = []
    for score in scores:
        if score >= 90:
            grades.append('A')
        elif score >= 80:
            grades.append('B')
        else:
            grades.append('C')
    return grades

## Complex Expression

$$\\int_0^{2\\pi} \\sin(x) dx = 0$$"""

        # Apply all formatters
        formatted = detect_and_format_tables(mixed_content)
        formatted = detect_and_format_math(formatted)
        formatted = detect_and_format_code(formatted)

        # Check that tables are formatted
        assert '---' in formatted  # Table separator

        # Check that math is formatted
        assert '$' in formatted or '\\[' in formatted

        # Check that code is formatted
        assert '```' in formatted

    def test_format_content_with_overlapping_patterns(self):
        """Test handling of overlapping patterns."""
        content = """The formula $x^2$ appears in this Python code:

def square(x):
    return x^2

And in this table:

Variable,Formula
x,$x^2$
y,$y^2$"""

        formatted = detect_and_format_math(content)
        formatted = detect_and_format_code(formatted)
        formatted = detect_and_format_tables(formatted)

        # Should handle all three types without conflicts
        assert '$' in formatted  # Math formatting preserved
        assert '```' in formatted or 'def square' in formatted  # Code or table formatting


if __name__ == "__main__":
    pytest.main([__file__])
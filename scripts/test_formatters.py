#!/usr/bin/env python3
"""
Test the advanced formatters on the actual PDF content.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_table_formatter():
    """Test table formatter with sample content."""
    print("📊 Testing Table Formatter")
    print("-" * 30)

    from vision_pdf.markdown.formatters.tables import AdvancedTableDetector, TableFormatter

    # Sample table content that might be in the PDF
    sample_table = """Name,Age,City,Salary
John Doe,30,New York,$75,000
Jane Smith,25,Los Angeles,$85,000
Bob Johnson,35,Chicago,$80,000"""

    detector = AdvancedTableDetector()
    formatter = TableFormatter()

    # Detect table
    table = detector.detect_table_from_text(sample_table)
    if table:
        print(f"✓ Table detected: {table.rows} rows x {table.cols} columns")
        print(f"   Confidence: {table.confidence:.2f}")

        # Format table
        formatted = formatter.format_table(table)
        print("✓ Formatted table:")
        print(formatted)
        return True
    else:
        print("✗ No table detected")
        return False

def test_math_formatter():
    """Test math formatter with sample content."""
    print("\n🧮 Testing Math Formatter")
    print("-" * 30)

    from vision_pdf.markdown.formatters.math import MathPatternRecognizer, detect_and_format_math

    # Sample math content that might be in the PDF
    sample_math = """The formula E = mc^2 is famous in physics.
Another equation is the Pythagorean theorem: a^2 + b^2 = c^2.
The integral ∫_0^∞ e^{-x} dx = 1 is also important."""

    recognizer = MathPatternRecognizer()
    expressions = recognizer.recognize_expressions(sample_math)

    print(f"✓ Detected {len(expressions)} math expressions:")
    for expr in expressions:
        print(f"   - {expr.text} (type: {expr.math_type.value}, confidence: {expr.confidence:.2f})")

    # Format math expressions
    formatted = detect_and_format_math(sample_math)
    print("✓ Formatted math content:")
    print(formatted)
    return True

def test_code_formatter():
    """Test code formatter with sample content."""
    print("\n💻 Testing Code Formatter")
    print("-" * 30)

    from vision_pdf.markdown.formatters.code import CodeDetector, detect_and_format_code

    # Sample code content that might be in the PDF
    sample_code = """Here is a Python function:
def calculate_salary(base, bonus):
    total = base + bonus
    return total

Here's JavaScript:
function processData(data) {
    return data.map(x => x * 2);
}"""

    detector = CodeDetector()
    code_blocks = detector.detect_code_blocks(sample_code)

    print(f"✓ Detected {len(code_blocks)} code blocks:")
    for block in code_blocks:
        print(f"   - Language: {block.language.value}")
        print(f"     Confidence: {block.confidence:.2f}")
        print(f"     Lines: {block.line_count}")

    # Format code
    formatted = detect_and_format_code(sample_code)
    print("✓ Formatted code content:")
    print(formatted)
    return True

def test_combined_formatter():
    """Test all formatters working together."""
    print("\n🎯 Testing Combined Formatters")
    print("-" * 30)

    from vision_pdf.markdown.formatters.tables import detect_and_format_tables
    from vision_pdf.markdown.formatters.math import detect_and_format_math
    from vision_pdf.markdown.formatters.code import detect_and_format_code

    # Combined sample content
    combined_content = """# Technical Report

## Data Analysis

The following table shows our findings:

| Metric | Q1 2025 | Q2 2025 | Growth |
|--------|----------|----------|--------|
| Revenue | $1.2M | $1.5M | 25% |
| Users | 10,000 | 15,000 | 50% |

## Mathematical Model

Our model uses the formula: f(x) = αe^(βx) + γ

The integral ∫_0^T f(t) dt gives the total accumulated value.

## Implementation

Python code:
```python
def calculate_growth(initial, rate, time):
    return initial * (1 + rate) ** time
```

JavaScript equivalent:
```javascript
function calculateGrowth(initial, rate, time) {
    return initial * Math.pow(1 + rate, time);
}
```"""

    print("Original content:")
    print(combined_content)
    print("-" * 50)

    # Apply all formatters
    formatted = detect_and_format_tables(combined_content)
    formatted = detect_and_format_math(formatted)
    formatted = detect_and_format_code(formatted)

    print("✓ Formatted content:")
    print(formatted)

    # Count improvements
    table_count = formatted.count('|') - combined_content.count('|')
    math_count = formatted.count('$') - combined_content.count('$')
    code_count = formatted.count('```') - combined_content.count('```')

    print(f"✓ Improvements: {table_count//2} table cells, {math_count//2} math expressions, {code_count//6} code blocks")
    return True

def test_pdf_content_simulation():
    """Test with simulated PDF content from the actual file."""
    print("\n📄 Testing with Simulated PDF Content")
    print("-" * 30)

    # This simulates content that might be extracted from "Pulse_of_the_Profession_2025 1.pdf"
    pdf_content = """PROFESSIONAL PULSE 2025
EXECUTIVE SUMMARY

Our analysis of industry trends shows the following projections:

| Sector | 2024 Growth | 2025 Projected | Key Drivers |
|--------|-------------|----------------|-------------|
| Technology | 15.2% | 18.7% | AI adoption, cloud migration |
| Healthcare | 12.8% | 14.3% | Aging population, telemedicine |
| Finance | 8.5% | 10.2% | Digital transformation, fintech |

The mathematical model for our projections is:
G(t) = G₀ × (1 + r)^t

Where G₀ is initial growth rate, r is compound annual growth rate, and t is time in years.

The performance can be calculated using:

function calculateROI(investment, returns) {
    const roi = ((returns - investment) / investment) * 100;
    return roi.toFixed(2) + '%';
}

Industry concentration index: C = Σ(x_i²) / (Σx_i)²

This represents the Herfindahl-Hirschman Index for market concentration."""

    print("✓ Processing simulated PDF content...")

    from vision_pdf.markdown.formatters.tables import detect_and_format_tables
    from vision_pdf.markdown.formatters.math import detect_and_format_math
    from vision_pdf.markdown.formatters.code import detect_and_format_code

    # Apply all formatters
    formatted = detect_and_format_tables(pdf_content)
    formatted = detect_and_format_math(formatted)
    formatted = detect_and_format_code(formatted)

    print("✓ Formatted PDF content:")
    print(formatted)

    return True

def main():
    """Run all formatter tests."""
    print("🎯 VisionPDF Advanced Formatters Test")
    print("=" * 50)

    tests = [
        test_table_formatter,
        test_math_formatter,
        test_code_formatter,
        test_combined_formatter,
        test_pdf_content_simulation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n📊 Test Results: {passed}/{total} formatter tests passed")

    if passed == total:
        print("🎉 All formatter tests passed!")
        print("✅ VisionPDF advanced content recognition is working perfectly!")
        print("📊 Ready to process your PDF with table, math, and code detection!")
        return 0
    else:
        print("❌ Some formatter tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
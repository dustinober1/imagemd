#!/usr/bin/env python3
"""
Test VisionPDF on the PMP Examination Content Outline PDF with focus on complex tables.
"""

import sys
import os
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pmp_pdf_tables():
    """Test processing the PMP PDF with emphasis on complex table detection."""
    print("🎯 Testing VisionPDF on PMP Examination Content Outline")
    print("📊 Focus: Complex Table Detection and Formatting")
    print("=" * 60)

    # Check if PDF file exists
    pdf_path = "New-PMP-Examination-Content-Outline-2026.pdf"
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return False

    file_size = Path(pdf_path).stat().st_size / 1024 / 1024
    print(f"📄 PDF file: {pdf_path} ({file_size:.1f} MB)")

    # Extract text from PDF
    print(f"\n🔍 Extracting text from PDF...")

    try:
        # Try PyPDF2 first
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"✅ PDF has {num_pages} pages")

            extracted_text = ""
            page_texts = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        page_texts.append(f"## Page {page_num}\n\n{page_text}")
                        extracted_text += f"\n\n=== PAGE {page_num} ===\n{page_text}\n"
                        print(f"   ✓ Extracted page {page_num} ({len(page_text)} characters)")
                    else:
                        print(f"   ⚠️  Page {page_num} appears to be empty or image-only")
                except Exception as e:
                    print(f"   ❌ Error extracting page {page_num}: {e}")
                    page_texts.append(f"## Page {page_num}\n\n[Error extracting text: {e}]")

            if not page_texts:
                print("❌ No text extracted from PDF")
                return False

    except Exception as e:
        print(f"❌ Failed to extract text from PDF: {e}")
        return False

    # Apply advanced formatters with emphasis on tables
    print(f"\n📊 Applying advanced formatters with table focus...")

    try:
        from vision_pdf.markdown.formatters.tables import AdvancedTableDetector, TableFormatter, detect_and_format_tables
        from vision_pdf.markdown.formatters.math import detect_and_format_math
        from vision_pdf.markdown.formatters.code import detect_and_format_code
        print("✅ All formatters imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import formatters: {e}")
        return False

    # Combine all page texts
    combined_text = "\n".join(page_texts)
    print(f"📝 Total extracted text: {len(combined_text)} characters")

    # Detailed table analysis before formatting
    print(f"\n🔍 Analyzing table structures...")

    table_detector = AdvancedTableDetector()
    table_formatter = TableFormatter()

    # Split into pages for individual analysis
    pages = combined_text.split("=== PAGE")
    table_summary = []
    total_tables = 0

    for page_idx, page_content in enumerate(pages[1:], 1):  # Skip first empty split
        if not page_content.strip():
            continue

        page_lines = page_content.split('\n')[2:]  # Remove page header
        page_text = '\n'.join(page_lines)

        # Detect tables in this page
        page_tables = []
        for i in range(0, len(page_lines), 10):  # Check in chunks of 10 lines
            chunk = page_lines[i:i+20]
            chunk_text = '\n'.join(chunk)
            if chunk_text.strip():
                table = table_detector.detect_table_from_text(chunk_text)
                if table and table.confidence > 0.5:
                    page_tables.append(table)

        if page_tables:
            table_summary.append({
                'page': page_idx,
                'tables': page_tables,
                'table_count': len(page_tables)
            })
            total_tables += len(page_tables)
            print(f"   📊 Page {page_idx}: {len(page_tables)} table(s) detected")

    print(f"\n📈 Table Detection Summary:")
    print(f"   Total pages with tables: {len(table_summary)}")
    print(f"   Total tables detected: {total_tables}")

    # Show details for each table
    for page_info in table_summary:
        print(f"\n   Page {page_info['page']} tables:")
        for i, table in enumerate(page_info['tables'], 1):
            print(f"     Table {i}: {table.rows}x{table.cols}, confidence={table.confidence:.2f}")
            if table.has_header:
                print(f"       Header: {table.headers[:3]}{'...' if len(table.headers) > 3 else ''}")

    # Apply formatters step by step
    start_time = time.time()

    print(f"\n📊 Formatting tables...")
    formatted_text = detect_and_format_tables(combined_text)

    print(f"🧮 Formatting mathematical expressions...")
    formatted_text = detect_and_format_math(formatted_text)

    print(f"💻 Formatting code blocks...")
    formatted_text = detect_and_format_code(formatted_text)

    end_time = time.time()
    processing_time = end_time - start_time

    # Analyze results
    print(f"\n⚡ Processing completed in {processing_time:.2f} seconds")

    # Count improvements
    original_lines = len(combined_text.split('\n'))
    formatted_lines = len(formatted_text.split('\n'))

    table_cells = formatted_text.count('|')
    math_expressions = formatted_text.count('$')
    code_blocks = formatted_text.count('```')

    print(f"\n📊 Content Analysis:")
    print(f"   Original lines: {original_lines}")
    print(f"   Formatted lines: {formatted_lines}")
    print(f"   Table cells: {table_cells}")
    print(f"   Math expressions: {math_expressions}")
    print(f"   Code blocks: {code_blocks // 6}")

    # Save the result
    output_file = "PMP_Examination_Content_2026_formatted.md"

    # Add comprehensive header
    header = f"""# PMP Examination Content Outline 2026 - Formatted with VisionPDF

**PDF file:** {pdf_path}
**File size:** {file_size:.1f} MB
**Pages processed:** {num_pages}
**Processing time:** {processing_time:.2f} seconds
**Total characters:** {len(formatted_text)}
**Tables detected:** {total_tables}
**Table cells formatted:** {table_cells}

---

## Table Detection Summary

This document contains {total_tables} tables detected across {len(table_summary)} pages.
VisionPDF advanced table detection has identified and formatted the following table structures:

"""

    # Add detailed table summary
    for page_info in table_summary:
        header += f"### Page {page_info['page']}\n\n"
        for i, table in enumerate(page_info['tables'], 1):
            header += f"- **Table {i}**: {table.rows} rows × {table.cols} columns"
            header += f" (confidence: {table.confidence:.2f})\n"
            if table.has_header and table.headers:
                header += f"  - Headers: {', '.join(table.headers[:5])}"
                if len(table.headers) > 5:
                    header += "..."
                header += "\n"
            header += f"  - Table type: {table.table_type.value}\n"
        header += "\n"

    header += "---\n\n"

    full_output = header + formatted_text

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)

        print(f"\n✅ Successfully saved formatted markdown to: {output_file}")
        output_size = Path(output_file).stat().st_size / 1024
        print(f"📁 Output file size: {output_size:.1f} KB")

        # Show preview of formatted tables
        print(f"\n📋 Preview of formatted tables:")
        print("-" * 70)

        preview_lines = formatted_text.split('\n')
        in_table = False
        table_count = 0
        lines_shown = 0

        for line in preview_lines:
            if '|' in line and ('|' in line.split('|', 1)[1] or line.strip().startswith('|')):
                if not in_table:
                    if table_count > 0:
                        print("-" * 70)
                    print(f"\n📊 Table {table_count + 1}:")
                    in_table = True
                    table_count += 1
                print(f"   {line}")
                lines_shown += 1
            elif in_table and not line.strip():
                print(f"   {line}")
                lines_shown += 1
            else:
                in_table = False

            if lines_shown >= 50:  # Limit preview
                break

        if table_count == 0:
            print("No table structures found in preview")
        else:
            print(f"\n✅ Previewed {table_count} table(s)")

        print("-" * 70)

        return True

    except Exception as e:
        print(f"❌ Failed to save output file: {e}")
        return False

def main():
    """Main function to run PMP PDF processing test."""
    print("🎯 VisionPDF PMP Complex Tables Test")
    print("=" * 50)

    # Test with PMP PDF file
    success = test_pmp_pdf_tables()

    if success:
        print("\n🎉 PMP PDF processing completed successfully!")
        print("✅ Advanced table detection has processed complex table structures")
        print("📊 Tables have been identified, analyzed, and formatted")
        print("📄 Check the output .md file for the complete formatted result")
        return 0
    else:
        print(f"\n❌ PMP PDF processing failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
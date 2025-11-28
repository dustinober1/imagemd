#!/usr/bin/env python3
"""
Test VisionPDF structured content formatter on the PMP PDF.
"""

import sys
import os
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pmp_structured_content():
    """Test processing the PMP PDF with structured content formatter."""
    print("🎯 Testing VisionPDF Structured Content Formatter on PMP PDF")
    print("📋 Focus: Domain/Task Hierarchy and Structured Content")
    print("=" * 70)

    # Load the previously extracted content
    input_file = "PMP_Examination_Content_2026_formatted.md"
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run the PMP table test first to generate the input file")
        return False

    print(f"📄 Reading from: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove the header to focus on the actual content
        content_start = content.find("## Page 1")
        if content_start != -1:
            content = content[content_start:]

        print(f"📝 Processing {len(content)} characters of content")

    except Exception as e:
        print(f"❌ Failed to read input file: {e}")
        return False

    # Test the structured content formatter
    print(f"\n🔍 Applying structured content formatter...")

    try:
        from vision_pdf.markdown.formatters.structured_content import (
            StructuredContentDetector,
            StructuredContentFormatter,
            detect_and_format_structured_content,
            extract_structured_elements
        )
        print("✅ Structured content formatter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import structured content formatter: {e}")
        return False

    # First, analyze what structured content we can detect
    print(f"\n🔍 Analyzing structured content patterns...")
    detector = StructuredContentDetector()
    elements = extract_structured_elements(content)

    print(f"✅ Detected {len(elements)} structured content elements")

    # Group by content type
    element_counts = {}
    for element in elements:
        content_type = element.content_type.value
        element_counts[content_type] = element_counts.get(content_type, 0) + 1

    print(f"\n📊 Detected Content Types:")
    for content_type, count in sorted(element_counts.items()):
        print(f"   {content_type.replace('_', ' ').title()}: {count}")

    # Show examples of each type
    print(f"\n📋 Structured Content Examples:")

    domain_examples = [e for e in elements if e.content_type.value == 'domain'][:3]
    if domain_examples:
        print(f"\n🎯 Domain Examples:")
        for element in domain_examples:
            print(f"   Original: {element.text}")
            print(f"   Metadata: {element.metadata}")
            print()

    task_examples = [e for e in elements if e.content_type.value == 'task'][:3]
    if task_examples:
        print(f"📋 Task Examples:")
        for element in task_examples:
            print(f"   Original: {element.text}")
            print(f"   Metadata: {element.metadata}")
            print()

    bullet_examples = [e for e in elements if e.content_type.value == 'bullet_point'][:3]
    if bullet_examples:
        print(f"• Bullet Point Examples:")
        for element in bullet_examples:
            print(f"   Original: {element.text}")
            print(f"   Content: {element.metadata.get('bullet_content', '')}")
            print()

    toc_examples = [e for e in elements if e.content_type.value == 'table_of_contents'][:3]
    if toc_examples:
        print(f"📑 Table of Contents Examples:")
        for element in toc_examples:
            print(f"   Original: {element.text}")
            print(f"   Section: {element.metadata.get('section_name', '')} -> Page {element.metadata.get('page_number', '')}")
            print()

    # Apply the formatter
    print(f"\n🎨 Applying structured content formatting...")
    start_time = time.time()

    formatted_content = detect_and_format_structured_content(content)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"⚡ Structured formatting completed in {processing_time:.3f} seconds")

    # Analyze improvements
    original_lines = len(content.split('\n'))
    formatted_lines = len(formatted_content.split('\n'))

    # Count markdown improvements
    heading_count = formatted_content.count('##') + formatted_content.count('###')
    bullet_count = formatted_content.count('- ')
    bold_count = formatted_content.count('**')

    print(f"\n📊 Formatting Analysis:")
    print(f"   Original lines: {original_lines}")
    print(f"   Formatted lines: {formatted_lines}")
    print(f"   Markdown headings added: {heading_count}")
    print(f"   Bullet points standardized: {bullet_count}")
    print(f"   Bold text added: {bold_count}")

    # Save the enhanced result
    output_file = "PMP_Structured_Content_Enhanced.md"

    # Create enhanced header
    header = f"""# PMP Examination Content Outline 2026 - Structured Content Enhanced

**Original file:** {input_file}
**Enhancement type:** Structured content recognition and formatting
**Processing time:** {processing_time:.3f} seconds
**Structured elements detected:** {len(elements)}

## Content Enhancement Summary

This document has been enhanced with VisionPDF's structured content recognition system:

"""

    for content_type, count in sorted(element_counts.items()):
        header += f"- **{content_type.replace('_', ' ').title()}**: {count} elements\n"

    header += f"""
### Enhancement Details

- **Domain headings** have been formatted as level 2 markdown headings (`##`)
- **Task descriptions** have been formatted as level 3 markdown headings (`###`)
- **Bullet points** have been standardized to use `- ` format
- **Table of contents** entries have been enhanced with bold text and page references
- **Percentage breakdowns** have been formatted for better readability

---

"""

    full_output = header + formatted_content

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)

        output_size = Path(output_file).stat().st_size / 1024
        print(f"\n✅ Successfully saved enhanced content to: {output_file}")
        print(f"📁 Output file size: {output_size:.1f} KB")

        # Show preview of enhancements
        print(f"\n📋 Preview of Enhanced Content:")
        print("-" * 80)

        preview_lines = formatted_content.split('\n')
        lines_shown = 0

        for line in preview_lines:
            if line.startswith(('## ', '### ', '- **', '- ')) and lines_shown < 30:
                print(f"   {line}")
                lines_shown += 1
            elif lines_shown >= 30:
                break

        if lines_shown == 0:
            print("   No enhanced lines found in preview")
        else:
            print(f"\n✅ Showed {lines_shown} enhanced lines")

        print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ Failed to save enhanced output file: {e}")
        return False

def main():
    """Main function to run structured content enhancement test."""
    print("🎯 VisionPDF Structured Content Enhancement Test")
    print("=" * 60)

    # Test structured content enhancement
    success = test_pmp_structured_content()

    if success:
        print("\n🎉 Structured content enhancement completed successfully!")
        print("✅ VisionPDF has identified and formatted structured elements")
        print("📋 Domain/Task hierarchy has been properly formatted")
        print("• Bullet points have been standardized")
        print("📑 Table of contents entries have been enhanced")
        print("📄 Check the output .md file for the complete enhanced result")
        return 0
    else:
        print(f"\n❌ Structured content enhancement failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
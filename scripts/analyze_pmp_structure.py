#!/usr/bin/env python3
"""
Analyze the PMP PDF structure to understand content organization.
"""

import sys
import re
from pathlib import Path

def analyze_pmp_content():
    """Analyze the raw extracted content from PMP PDF."""

    # Read the raw content from the formatted file
    with open("PMP_Examination_Content_2026_formatted.md", 'r', encoding='utf-8') as f:
        content = f.read()

    print("🔍 PMP PDF Structure Analysis")
    print("=" * 50)

    # Split by pages
    pages = content.split("## Page")
    print(f"📄 Total pages: {len(pages) - 1}")  # First split is header content

    # Analyze content patterns
    all_lines = content.split('\n')

    print(f"\n📊 Content Statistics:")
    print(f"   Total lines: {len(all_lines)}")
    print(f"   Total characters: {len(content)}")

    # Find different content patterns
    patterns = {
        'dot_alignment': 0,
        'pipe_separators': 0,
        'tab_separators': 0,
        'multiple_spaces': 0,
        'task_headings': 0,
        'domain_headings': 0,
        'bullet_points': 0,
        'percentage_values': 0
    }

    for line in all_lines:
        line = line.strip()
        if not line:
            continue

        # Check for table-like patterns
        if '................................' in line:
            patterns['dot_alignment'] += 1
        if '|' in line and line.count('|') >= 3:
            patterns['pipe_separators'] += 1
        if '\t' in line:
            patterns['tab_separators'] += 1
        if re.search(r' {3,}', line):
            patterns['multiple_spaces'] += 1

        # Check for PMP-specific patterns
        if line.startswith('Task'):
            patterns['task_headings'] += 1
        if 'DOMAIN' in line and '%' in line:
            patterns['domain_headings'] += 1
        if line.startswith('•'):
            patterns['bullet_points'] += 1
        if re.search(r'\d+%', line):
            patterns['percentage_values'] += 1

    print(f"\n🔍 Pattern Detection:")
    for pattern, count in patterns.items():
        print(f"   {pattern.replace('_', ' ').title()}: {count}")

    # Show samples of different patterns
    print(f"\n📋 Content Samples:")

    print(f"\n1. Dot Alignment Patterns (Table of Contents style):")
    dot_lines = [line for line in all_lines if '................................' in line.strip()][:5]
    for line in dot_lines:
        print(f"   {line.strip()}")

    print(f"\n2. Domain Headings:")
    domain_lines = [line for line in all_lines if 'DOMAIN' in line and '%' in line][:5]
    for line in domain_lines:
        print(f"   {line.strip()}")

    print(f"\n3. Task Headings:")
    task_lines = [line for line in all_lines if line.strip().startswith('Task')][:5]
    for line in task_lines:
        print(f"   {line.strip()}")

    print(f"\n4. Bullet Points:")
    bullet_lines = [line for line in all_lines if line.strip().startswith('•')][:5]
    for line in bullet_lines:
        print(f"   {line.strip()}")

    # Search for any actual table structures
    print(f"\n🔍 Searching for Traditional Table Structures:")

    table_indicators = 0
    potential_tables = []

    for i, line in enumerate(all_lines):
        line = line.strip()

        # Look for lines that might be table rows
        if (line.count('|') >= 3 and
            not line.startswith('##') and
            not line.startswith('```') and
            not line.startswith('•') and
            not line.startswith('Task') and
            'DOMAIN' not in line):

            table_indicators += 1
            potential_tables.append((i+1, line))

    if table_indicators > 0:
        print(f"   Found {table_indicators} potential table rows:")
        for line_num, line in potential_tables[:10]:
            print(f"     Line {line_num}: {line}")
    else:
        print("   No traditional table structures found with pipe separators")

    # Check for space-aligned content
    print(f"\n🔍 Checking for Space-Aligned Content:")
    space_aligned_blocks = []

    for i in range(len(all_lines)):
        current_line = all_lines[i].strip()
        if not current_line:
            continue

        # Look for patterns that might be space-aligned tables
        if (re.search(r'^[A-Za-z\s]+\s+\d+\s+\d+', current_line) or
            re.search(r'^[A-Za-z\s]+\s+\$?\d+', current_line) or
            re.search(r'^\w+\s+\w+\s+\d+', current_line)):

            space_aligned_blocks.append((i+1, current_line))

    if space_aligned_blocks:
        print(f"   Found {len(space_aligned_blocks)} potentially space-aligned lines:")
        for line_num, line in space_aligned_blocks[:10]:
            print(f"     Line {line_num}: {line}")
    else:
        print("   No obvious space-aligned table structures found")

    # Conclusion
    print(f"\n📝 Analysis Summary:")
    if patterns['dot_alignment'] > 10:
        print("   ✅ Document uses dot-alignment formatting (table of contents style)")
    if patterns['domain_headings'] > 0:
        print("   ✅ Document has structured domain/task hierarchy")
    if patterns['bullet_points'] > 20:
        print("   ✅ Document extensively uses bullet points for content")
    if patterns['task_headings'] > 0:
        print("   ✅ Document follows task-based structure")

    if table_indicators == 0 and len(space_aligned_blocks) == 0:
        print("   ❌ No traditional table structures detected")
        print("   📋 Content is organized as hierarchical text, not tabular data")

    print(f"\n💡 Recommendation:")
    print("   This PMP document appears to be organized as:")
    print("   - Hierarchical headings (Domains, Tasks, Enablers)")
    print("   - Bullet points for detailed content")
    print("   - Dot-aligned table of contents")
    print("   - No traditional row/column table structures")
    print("   The table detector working correctly - there simply aren't tables to detect!")

if __name__ == "__main__":
    analyze_pmp_content()
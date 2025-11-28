"""
Structured content formatter for VisionPDF.

This module provides specialized formatting for structured content like
hierarchical documents, domain/task structures, and table of contents.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ContentType(Enum):
    """Types of structured content."""
    DOMAIN = "domain"
    TASK = "task"
    BULLET_POINT = "bullet"
    TABLE_OF_CONTENTS = "table_of_contents"
    HIERARCHICAL = "hierarchical"
    PERCENTAGE_BREAKDOWN = "percentage_breakdown"


@dataclass
class StructuredElement:
    """Represents a structured content element."""
    content_type: ContentType
    text: str
    level: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize structured element."""
        if self.metadata is None:
            self.metadata = {}


class StructuredContentDetector:
    """Detect structured content patterns in documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured content detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Compile patterns for different content types
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[ContentType, re.Pattern]:
        """Compile patterns for structured content detection."""
        patterns = {}

        # Domain patterns (e.g., "DOMAIN I PEOPLE —33%")
        patterns[ContentType.DOMAIN] = re.compile(
            r'^\s*(?:DOMAIN\s+([IVXLCDM]+)\s+|Project\s+Management\s+Professional.*DOMAIN\s+([IVXLCDM]+)\s+)([^—]+?)\s*—\s*(\d+%)',
            re.IGNORECASE
        )

        # Task patterns (e.g., "Task 1 Develop a common vision")
        patterns[ContentType.TASK] = re.compile(
            r'^\s*(?:Task\s+(\d+)|Task\s+statement\s*:)\s*(.+)?$',
            re.IGNORECASE
        )

        # Bullet point patterns
        patterns[ContentType.BULLET_POINT] = re.compile(
            r'^\s*[•\-\*]\s*(.+)$'
        )

        # Table of contents patterns (dot alignment)
        patterns[ContentType.TABLE_OF_CONTENTS] = re.compile(
            r'^\s*([A-Z][^\.]{10,50})\s*\.+\s*(\d+)\s*$',
            re.IGNORECASE
        )

        # Percentage breakdown patterns
        patterns[ContentType.PERCENTAGE_BREAKDOWN] = re.compile(
            r'^\s*([^:]+?)\s*[:\-]?\s*(\d+%)\s*$'
        )

        return patterns

    def detect_structured_content(self, text: str) -> List[StructuredElement]:
        """
        Detect structured content elements in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected structured elements
        """
        lines = text.split('\n')
        elements = []

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check each content type
            for content_type, pattern in self.patterns.items():
                match = pattern.search(line_stripped)
                if match:
                    element = self._create_element(content_type, line_stripped, match, line_num)
                    if element:
                        elements.append(element)
                    break  # Only match one pattern per line

        return elements

    def _create_element(self, content_type: ContentType, text: str, match: re.Match, line_num: int) -> Optional[StructuredElement]:
        """Create a structured element from a match."""
        try:
            if content_type == ContentType.DOMAIN:
                domain_num = match.group(1) or match.group(2)
                domain_name = match.group(3) if match.group(3) else (match.group(4) if match.group(4) else "")
                percentage = match.group(4) if match.group(4) else (match.group(5) if len(match.groups()) > 4 else "")

                return StructuredElement(
                    content_type=content_type,
                    text=text,
                    level=1,
                    metadata={
                        'domain_number': domain_num,
                        'domain_name': domain_name.strip(),
                        'percentage': percentage,
                        'line_number': line_num
                    }
                )

            elif content_type == ContentType.TASK:
                task_num = match.group(1)
                task_desc = match.group(2) if match.group(2) else ""

                return StructuredElement(
                    content_type=content_type,
                    text=text,
                    level=2,
                    metadata={
                        'task_number': task_num,
                        'task_description': task_desc.strip(),
                        'line_number': line_num
                    }
                )

            elif content_type == ContentType.BULLET_POINT:
                return StructuredElement(
                    content_type=content_type,
                    text=text,
                    level=3,
                    metadata={
                        'bullet_content': match.group(1).strip(),
                        'line_number': line_num
                    }
                )

            elif content_type == ContentType.TABLE_OF_CONTENTS:
                section_name = match.group(1).strip()
                page_number = match.group(2)

                return StructuredElement(
                    content_type=content_type,
                    text=text,
                    level=1,
                    metadata={
                        'section_name': section_name,
                        'page_number': page_number,
                        'line_number': line_num
                    }
                )

            elif content_type == ContentType.PERCENTAGE_BREAKDOWN:
                category = match.group(1).strip()
                percentage = match.group(2)

                return StructuredElement(
                    content_type=content_type,
                    text=text,
                    level=2,
                    metadata={
                        'category': category,
                        'percentage': percentage,
                        'line_number': line_num
                    }
                )

        except (IndexError, AttributeError) as e:
            logger.warning(f"Error creating element from match: {e}")
            return None

        return None


class StructuredContentFormatter:
    """Format structured content with enhanced markdown."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured content formatter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.detector = StructuredContentDetector(config)

    def format_structured_content(self, text: str) -> str:
        """
        Format text containing structured content.

        Args:
            text: Text to format

        Returns:
            Formatted text with enhanced markdown
        """
        # Detect structured elements
        elements = self.detector.detect_structured_content(text)

        if not elements:
            return text

        # Sort elements by line number to maintain order
        elements.sort(key=lambda x: x.metadata.get('line_number', 0))

        # Group elements by type for better formatting
        formatted_sections = self._format_by_sections(elements)

        # Replace original text with formatted sections
        lines = text.split('\n')
        processed_lines = set()

        # Apply formatting in reverse order to maintain line numbers
        for element in reversed(elements):
            line_num = element.metadata.get('line_number', 0)
            if line_num not in processed_lines and 0 <= line_num < len(lines):
                formatted_line = self._format_element(element)
                if formatted_line != lines[line_num]:
                    lines[line_num] = formatted_line
                    processed_lines.add(line_num)

        return '\n'.join(lines)

    def _format_element(self, element: StructuredElement) -> str:
        """Format an individual structured element."""
        if element.content_type == ContentType.DOMAIN:
            return self._format_domain(element)
        elif element.content_type == ContentType.TASK:
            return self._format_task(element)
        elif element.content_type == ContentType.BULLET_POINT:
            return self._format_bullet_point(element)
        elif element.content_type == ContentType.TABLE_OF_CONTENTS:
            return self._format_toc_entry(element)
        elif element.content_type == ContentType.PERCENTAGE_BREAKDOWN:
            return self._format_percentage_breakdown(element)
        else:
            return element.text

    def _format_domain(self, element: StructuredElement) -> str:
        """Format a domain element."""
        metadata = element.metadata
        domain_num = metadata.get('domain_number', '')
        domain_name = metadata.get('domain_name', '')
        percentage = metadata.get('percentage', '')

        # Create a nicely formatted domain heading
        if domain_num and domain_name and percentage:
            return f"## Domain {domain_num}: {domain_name} ({percentage})"
        elif domain_name and percentage:
            return f"## {domain_name} ({percentage})"
        else:
            return f"## {element.text}"

    def _format_task(self, element: StructuredElement) -> str:
        """Format a task element."""
        metadata = element.metadata
        task_num = metadata.get('task_number', '')
        task_desc = metadata.get('task_description', '')

        if task_num and task_desc:
            return f"### Task {task_num}: {task_desc}"
        elif task_desc:
            return f"### {task_desc}"
        else:
            return f"### {element.text}"

    def _format_bullet_point(self, element: StructuredElement) -> str:
        """Format a bullet point element."""
        content = element.metadata.get('bullet_content', element.text[1:].strip())
        return f"- {content}"

    def _format_toc_entry(self, element: StructuredElement) -> str:
        """Format a table of contents entry."""
        metadata = element.metadata
        section_name = metadata.get('section_name', '')
        page_number = metadata.get('page_number', '')

        if section_name and page_number:
            return f"- **{section_name}** ... Page {page_number}"
        else:
            return f"- {element.text}"

    def _format_percentage_breakdown(self, element: StructuredElement) -> str:
        """Format a percentage breakdown element."""
        metadata = element.metadata
        category = metadata.get('category', '')
        percentage = metadata.get('percentage', '')

        if category and percentage:
            return f"- **{category}**: {percentage}"
        else:
            return element.text

    def _format_by_sections(self, elements: List[StructuredElement]) -> str:
        """Group and format elements by logical sections."""
        # This could be expanded to create more complex section-based formatting
        sections = {}

        for element in elements:
            section_type = element.content_type.value
            if section_type not in sections:
                sections[section_type] = []
            sections[section_type].append(element)

        # For now, we'll handle this inline in format_structured_content
        return ""


def detect_and_format_structured_content(content: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect and format structured content in the given text.

    Args:
        content: Content to process
        config: Configuration dictionary

    Returns:
        Content with structured elements properly formatted
    """
    formatter = StructuredContentFormatter(config)
    return formatter.format_structured_content(content)


def extract_structured_elements(text: str, config: Optional[Dict[str, Any]] = None) -> List[StructuredElement]:
    """
    Extract structured elements from text.

    Args:
        text: Text to analyze
        config: Configuration dictionary

    Returns:
        List of detected structured elements
    """
    detector = StructuredContentDetector(config)
    return detector.detect_structured_content(text)
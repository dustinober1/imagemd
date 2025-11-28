"""
Mathematical expression recognition and LaTeX formatting for VisionPDF.

This module provides sophisticated mathematical expression detection,
LaTeX conversion, and formatting capabilities for preserving
mathematical content in PDF to markdown conversion.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

from ...core.document import ContentElement, ContentType
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class MathType(Enum):
    """Types of mathematical expressions."""
    INLINE = "inline"           # $...$
    DISPLAY = "display"         # $$...$$ or \[...\]
    FRACTION = "fraction"       # \frac{a}{b}
    SUPERSCRIPT = "superscript" # x^2
    SUBSCRIPT = "subscript"     # x_i
    INTEGRAL = "integral"       # \int...
    SUMMATION = "summation"     # \sum...
    MATRIX = "matrix"           # \begin{matrix}...\end{matrix}
    EQUATION = "equation"       # Complex equations
    GREEK = "greek"            # Greek letters
    SYMBOL = "symbol"           # Mathematical symbols


@dataclass
class MathExpression:
    """Represents a detected mathematical expression."""
    text: str
    math_type: MathType
    confidence: float
    latex: str
    position: Optional[Tuple[int, int]] = None  # (start, end) in original text
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize math expression."""
        if self.metadata is None:
            self.metadata = {}


class MathPatternRecognizer:
    """Recognize mathematical patterns in text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize math pattern recognizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

        # Compile regex patterns
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile mathematical expression patterns."""
        patterns = {
            # LaTeX patterns
            'latex_inline': re.compile(r'\$([^$]+)\$'),
            'latex_display': re.compile(r'\$\$([^$]+)\$\$'),
            'latex_inline_brackets': re.compile(r'\\\((.*?)\\\)'),
            'latex_display_brackets': re.compile(r'\\\[([^\\]+)\\\]'),

            # Mathematical operations
            'fraction': re.compile(r'(\w+)\s*/\s*(\w+)'),
            'superscript': re.compile(r'(\w+)\^(\{?\w+\}?)'),
            'subscript': re.compile(r'(\w+)_(\{?\w+\}?)'),

            # Mathematical symbols
            'integral': re.compile(r'∫|\\int'),
            'summation': re.compile(r'∑|\\sum'),
            'product': re.compile(r'∏|\\prod'),
            'partial_derivative': re.compile(r'∂|\\partial'),
            'nabla': re.compile(r'∇|\\nabla'),
            'infinity': re.compile(r'∞|\\infty'),

            # Greek letters
            'greek_letters': re.compile(r'[α-ωΑ-Ω]|\\[a-zA-Z]+'),

            # Mathematical operators
            'comparison': re.compile(r'[≤≥≠≈≡]|\\[lg]eq|\\neq|\\approx|\\equiv'),
            'arrows': re.compile(r'[→←↑↓↔]|\\[lr]arrow|\\leftrightarrow'),

            # Complex expressions
            'equation': re.compile(r'\w+\s*[=≠≤≥]\s*[\w\d\(\)\+\-\*/\^_]+'),
            'function': re.compile(r'\w+\s*\([^)]*\)'),
        }

        return patterns

    def recognize_expressions(self, text: str) -> List[MathExpression]:
        """
        Recognize mathematical expressions in text.

        Args:
            text: Text to analyze

        Returns:
            List of recognized math expressions
        """
        expressions = []

        # Check for existing LaTeX first (highest confidence)
        latex_expressions = self._find_latex_expressions(text)
        expressions.extend(latex_expressions)

        # Find other mathematical patterns
        math_expressions = self._find_mathematical_patterns(text)
        expressions.extend(math_expressions)

        # Sort by position and remove overlaps
        expressions = self._sort_and_deduplicate_expressions(expressions)

        return expressions

    def _find_latex_expressions(self, text: str) -> List[MathExpression]:
        """Find existing LaTeX expressions."""
        expressions = []

        # Check inline LaTeX $...$
        for match in self.patterns['latex_inline'].finditer(text):
            expression = MathExpression(
                text=match.group(1),
                math_type=MathType.INLINE,
                confidence=0.95,
                latex=f"${match.group(1)}$",
                position=match.span()
            )
            expressions.append(expression)

        # Check display LaTeX $$...$$
        for match in self.patterns['latex_display'].finditer(text):
            expression = MathExpression(
                text=match.group(1),
                math_type=MathType.DISPLAY,
                confidence=0.95,
                latex=f"$${match.group(1)}$$",
                position=match.span()
            )
            expressions.append(expression)

        # Check LaTeX inline brackets \(...)
        for match in self.patterns['latex_inline_brackets'].finditer(text):
            expression = MathExpression(
                text=match.group(1),
                math_type=MathType.INLINE,
                confidence=0.90,
                latex=f"\\({match.group(1)}\\)",
                position=match.span()
            )
            expressions.append(expression)

        # Check LaTeX display brackets \[...\]
        for match in self.patterns['latex_display_brackets'].finditer(text):
            expression = MathExpression(
                text=match.group(1),
                math_type=MathType.DISPLAY,
                confidence=0.90,
                latex=f"\\[{match.group(1)}\\]",
                position=match.span()
            )
            expressions.append(expression)

        return expressions

    def _find_mathematical_patterns(self, text: str) -> List[MathExpression]:
        """Find mathematical patterns that aren't already in LaTeX."""
        expressions = []

        # Find fractions a/b
        for match in self.patterns['fraction'].finditer(text):
            # Skip if it's part of a larger LaTeX expression
            if self._is_in_latex_context(match.start(), text):
                continue

            numerator = match.group(1)
            denominator = match.group(2)
            latex = f"\\frac{{{numerator}}}{{{denominator}}}"

            expression = MathExpression(
                text=match.group(0),
                math_type=MathType.FRACTION,
                confidence=0.80,
                latex=latex,
                position=match.span()
            )
            expressions.append(expression)

        # Find superscripts
        for match in self.patterns['superscript'].finditer(text):
            if self._is_in_latex_context(match.start(), text):
                continue

            base = match.group(1)
            power = match.group(2).strip('{}')
            latex = f"{base}^{{{power}}}"

            expression = MathExpression(
                text=match.group(0),
                math_type=MathType.SUPERSCRIPT,
                confidence=0.75,
                latex=latex,
                position=match.span()
            )
            expressions.append(expression)

        # Find subscripts
        for match in self.patterns['subscript'].finditer(text):
            if self._is_in_latex_context(match.start(), text):
                continue

            base = match.group(1)
            index = match.group(2).strip('{}')
            latex = f"{base}_{{{index}}}"

            expression = MathExpression(
                text=match.group(0),
                math_type=MathType.SUBSCRIPT,
                confidence=0.75,
                latex=latex,
                position=match.span()
            )
            expressions.append(expression)

        # Find integrals
        for match in self.patterns['integral'].finditer(text):
            if self._is_in_latex_context(match.start(), text):
                continue

            # Simple conversion for integrals
            if '∫' in match.group(0):
                latex = match.group(0).replace('∫', '\\int')
            else:
                latex = match.group(0)

            expression = MathExpression(
                text=match.group(0),
                math_type=MathType.INTEGRAL,
                confidence=0.70,
                latex=latex,
                position=match.span()
            )
            expressions.append(expression)

        # Find complex equations
        for match in self.patterns['equation'].finditer(text):
            if self._is_in_latex_context(match.start(), text):
                continue

            equation_text = match.group(0)
            latex = self._convert_equation_to_latex(equation_text)

            expression = MathExpression(
                text=equation_text,
                math_type=MathType.EQUATION,
                confidence=0.65,
                latex=latex,
                position=match.span()
            )
            expressions.append(expression)

        return expressions

    def _is_in_latex_context(self, pos: int, text: str) -> bool:
        """Check if a position is within a LaTeX expression."""
        # Check for $ delimiters
        dollar_positions = [m.start() for m in re.finditer(r'\$', text)]

        if len(dollar_positions) >= 2:
            for i in range(0, len(dollar_positions), 2):
                if i + 1 < len(dollar_positions):
                    start = dollar_positions[i]
                    end = dollar_positions[i + 1]
                    if start < pos < end:
                        return True

        return False

    def _convert_equation_to_latex(self, equation: str) -> str:
        """Convert a plain text equation to LaTeX."""
        latex = equation

        # Convert common mathematical symbols
        replacements = {
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '≈': '\\approx',
            '∞': '\\infty',
            'π': '\\pi',
            '√': '\\sqrt{}',
        }

        for symbol, replacement in replacements.items():
            latex = latex.replace(symbol, replacement)

        # Fix fractions that weren't caught earlier
        latex = re.sub(r'(\w+)\s*/\s*(\w+)', r'\\frac{\1}{\2}', latex)

        # Fix multiplication
        latex = re.sub(r'(\d)([a-zA-Z])', r'\1 \\cdot \2', latex)

        return latex

    def _sort_and_deduplicate_expressions(self, expressions: List[MathExpression]) -> List[MathExpression]:
        """Sort expressions by position and remove duplicates."""
        if not expressions:
            return []

        # Sort by position
        expressions.sort(key=lambda x: x.position[0] if x.position else 0)

        # Remove overlapping expressions
        filtered = []
        for expr in expressions:
            if not expr.position:
                filtered.append(expr)
                continue

            # Check for overlap with existing expressions
            overlap = False
            for existing in filtered:
                if (existing.position and
                    expr.position[0] >= existing.position[0] and
                    expr.position[1] <= existing.position[1]):
                    overlap = True
                    break

            if not overlap:
                filtered.append(expr)

        return filtered


class MathFormatter:
    """Format mathematical expressions in markdown."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize math formatter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_display_math_for_complex = self.config.get('use_display_math_for_complex', True)
        self.complexity_threshold = self.config.get('complexity_threshold', 20)

    def format_text_with_math(self, text: str) -> str:
        """
        Format text containing mathematical expressions.

        Args:
            text: Text to format

        Returns:
            Text with math expressions properly formatted
        """
        recognizer = MathPatternRecognizer(self.config)
        expressions = recognizer.recognize_expressions(text)

        if not expressions:
            return text

        # Sort expressions by position (reverse order to maintain indices)
        expressions.sort(key=lambda x: x.position[0] if x.position else 0, reverse=True)

        # Replace each expression with formatted LaTeX
        formatted_text = text
        for expr in expressions:
            if expr.position:
                start, end = expr.position
                latex = self._format_expression(expr)
                formatted_text = formatted_text[:start] + latex + formatted_text[end:]

        return formatted_text

    def _format_expression(self, expression: MathExpression) -> str:
        """Format a single mathematical expression."""
        latex = expression.latex

        # Use display math for complex expressions
        if (self.use_display_math_for_complex and
            len(expression.text) > self.complexity_threshold and
            expression.math_type in [MathType.EQUATION, MathType.INTEGRAL, MathType.SUMMATION]):

            # Convert inline to display format
            if latex.startswith('$') and latex.endswith('$'):
                latex = latex[1:-1]
                latex = f"$${latex}$$"
            elif latex.startswith('\\(') and latex.endswith('\\)'):
                latex = latex[2:-2]
                latex = f"\\[{latex}\\]"

        return latex

    def format_math_element(self, element: ContentElement) -> str:
        """
        Format a math content element.

        Args:
            element: Math content element

        Returns:
            Formatted mathematical content
        """
        return self.format_text_with_math(element.text)


def detect_and_format_math(content: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect and format mathematical expressions in content.

    Args:
        content: Content to process
        config: Configuration dictionary

    Returns:
        Content with math expressions formatted as LaTeX
    """
    formatter = MathFormatter(config)
    return formatter.format_text_with_math(content)


def extract_math_expressions(text: str) -> List[MathExpression]:
    """
    Extract mathematical expressions from text.

    Args:
        text: Text to analyze

    Returns:
        List of detected mathematical expressions
    """
    recognizer = MathPatternRecognizer()
    return recognizer.recognize_expressions(text)


def convert_to_latex(expression: str) -> str:
    """
    Convert a mathematical expression to LaTeX.

    Args:
        expression: Mathematical expression

    Returns:
        LaTeX representation
    """
    recognizer = MathPatternRecognizer()
    expressions = recognizer.recognize_expressions(expression)

    if expressions:
        # Return the LaTeX of the first (highest confidence) expression
        return expressions[0].latex
    else:
        # Try to convert as plain equation
        latex = expression

        # Basic conversions
        replacements = {
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '≈': '\\approx',
            '∞': '\\infty',
            'π': '\\pi',
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'θ': '\\theta',
            'λ': '\\lambda',
            'μ': '\\mu',
            'σ': '\\sigma',
            'φ': '\\phi',
            'ω': '\\omega',
            'Δ': '\\Delta',
            'Σ': '\\Sigma',
            'Π': '\\Pi',
            'Ω': '\\Omega',
            '∫': '\\int',
            '∑': '\\sum',
            '∏': '\\prod',
            '∂': '\\partial',
            '∇': '\\nabla',
        }

        for symbol, replacement in replacements.items():
            latex = latex.replace(symbol, replacement)

        # Wrap in inline math if not already
        if not latex.startswith('$') and not latex.startswith('\\'):
            latex = f"${latex}$"

        return latex


def validate_latex(latex: str) -> bool:
    """
    Validate if LaTeX syntax is likely correct.

    Args:
        latex: LaTeX string to validate

    Returns:
        True if LaTeX appears syntactically correct
    """
    # Basic validation checks
    if not latex:
        return False

    # Check for balanced delimiters
    delimiter_pairs = [
        ('$', '$'),
        ('$$', '$$'),
        (r'\(', r'\)'),
        (r'\[', r'\]'),
        ('{', '}'),
    ]

    for open_delim, close_delim in delimiter_pairs:
        open_count = latex.count(open_delim)
        close_count = latex.count(close_delim)
        if open_count != close_count:
            return False

    # Check for invalid characters in math mode
    invalid_patterns = [
        r'[^\\]\$',  # Unescaped dollar sign
        r'\\\\$',    # Double backslash at end
    ]

    for pattern in invalid_patterns:
        if re.search(pattern, latex):
            return False

    return True
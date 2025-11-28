"""
Code block identification and syntax highlighting for VisionPDF.

This module provides sophisticated code detection, language identification,
and syntax-highlighted markdown formatting for code blocks.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

from ...core.document import ContentElement, ContentType
from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    MARKDOWN = "markdown"
    DOCKERFILE = "dockerfile"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Represents a detected code block."""
    lines: List[str]
    language: CodeLanguage
    confidence: float
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize code block."""
        if self.metadata is None:
            self.metadata = {}

    @property
    def text(self) -> str:
        """Get the full text of the code block."""
        return '\n'.join(self.lines)

    @property
    def line_count(self) -> int:
        """Get the number of lines in the code block."""
        return len(self.lines)


class LanguageDetector:
    """Detect programming languages in code blocks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize language detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

        # Compile language patterns
        self.patterns = self._compile_language_patterns()

    def _compile_language_patterns(self) -> Dict[CodeLanguage, Dict[str, re.Pattern]]:
        """Compile regex patterns for language detection."""
        patterns = {}

        # Python patterns
        patterns[CodeLanguage.PYTHON] = {
            'keywords': re.compile(r'\b(def|class|import|from|if __name__|elif|try|except|finally|with|yield|lambda|async|await)\b'),
            'builtins': re.compile(r'\b(print|len|range|list|dict|str|int|float|bool|set|tuple|open|input|type|isinstance|super)\b'),
            'comments': re.compile(r'#.*$'),
            'indentation': re.compile(r'^(\s+)'),
            'functions': re.compile(r'\w+\s*\([^)]*\)\s*:'),
        }

        # JavaScript patterns
        patterns[CodeLanguage.JAVASCRIPT] = {
            'keywords': re.compile(r'\b(function|var|let|const|if|else|for|while|do|switch|case|break|continue|return|try|catch|finally|throw|new|this|typeof|instanceof)\b'),
            'builtins': re.compile(r'\b(console|document|window|Array|Object|String|Number|Boolean|Date|RegExp|Math|JSON)\b'),
            'comments': re.compile(r'//.*$|/\*[\s\S]*?\*/'),
            'arrows': re.compile(r'\s*=>\s*'),
        }

        # Java patterns
        patterns[CodeLanguage.JAVA] = {
            'keywords': re.compile(r'\b(public|private|protected|static|final|abstract|class|interface|extends|implements|import|package|void|int|String|boolean)\b'),
            'annotations': re.compile(r'@\w+'),
            'methods': re.compile(r'\b(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*(throws\s+\w+\s*)?\{'),
        }

        # C/C++ patterns
        patterns[CodeLanguage.C] = {
            'keywords': re.compile(r'\b(include|stdio|stdlib|string|math|int|char|float|double|void|if|else|for|while|do|switch|case|break|continue|return|struct|typedef|enum)\b'),
            'includes': re.compile(r'#include\s*[<"][^>"]+[>"]'),
            'pointers': re.compile(r'\*\w+|\w+\*'),
            'functions': re.compile(r'\w+\s*\([^)]*\)\s*;'),
        }

        # C# patterns
        patterns[CodeLanguage.CSHARP] = {
            'keywords': re.compile(r'\b(public|private|protected|internal|static|readonly|const|virtual|override|abstract|sealed|class|interface|namespace|using|var|void|int|string|bool|double)\b'),
            'properties': re.compile(r'\w+\s*\{\s*get;\s*set;\s*\}'),
            'linq': re.compile(r'\b(from|where|select|orderby|group|join)\b'),
        }

        # PHP patterns
        patterns[CodeLanguage.PHP] = {
            'tags': re.compile(r'<\?php|\?>'),
            'keywords': re.compile(r'\b(function|class|if|else|elseif|for|foreach|while|do|switch|case|break|continue|return|try|catch|finally|throw|new|this|\$|\w+)\b'),
            'variables': re.compile(r'\$\w+'),
            'arrays': re.compile(r'array\s*\(|\[.*?\]'),
        }

        # Ruby patterns
        patterns[CodeLanguage.RUBY] = {
            'keywords': re.compile(r'\b(def|class|module|if|unless|else|elsif|case|when|while|until|for|do|break|next|return|yield|begin|rescue|ensure|end|require|include)\b'),
            'symbols': re.compile(r':\w+'),
            'blocks': re.compile(r'\{.*?\}|do\s+.*?\s+end'),
            'interpolation': re.compile(r'#\{.*?\}'),
        }

        # Go patterns
        patterns[CodeLanguage.GO] = {
            'keywords': re.compile(r'\b(func|package|import|type|struct|interface|var|const|if|else|for|switch|case|default|break|continue|return|go|select|defer|chan)\b'),
            'types': re.compile(r'\b(int|string|bool|float64|error|map|slice)\b'),
            'functions': re.compile(r'func\s+\w+\s*\([^)]*\)'),
        }

        # Rust patterns
        patterns[CodeLanguage.RUST] = {
            'keywords': re.compile(r'\b(fn|let|mut|const|static|if|else|match|for|while|loop|break|continue|return|struct|enum|impl|trait|mod|use|pub|crate|super)\b'),
            'lifetimes': re.compile(r"'\w+"),
            'patterns': re.compile(r'Some\(|None|Ok\(|Err\(|Result<'),
            'macros': re.compile(r'\w+\s*!'),
        }

        # SQL patterns
        patterns[CodeLanguage.SQL] = {
            'keywords': re.compile(r'\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|TABLE|ALTER|DROP|INDEX|JOIN|INNER|LEFT|RIGHT|OUTER|GROUP|BY|ORDER|HAVING|UNION|DISTINCT|COUNT|SUM|AVG|MAX|MIN)\b', re.IGNORECASE),
            'functions': re.compile(r'\b(COUNT|SUM|AVG|MAX|MIN|UPPER|LOWER|SUBSTRING|CONCAT|COALESCE|CAST)\b', re.IGNORECASE),
            'clauses': re.compile(r'\b(AND|OR|NOT|IN|EXISTS|BETWEEN|LIKE|IS NULL|IS NOT NULL)\b', re.IGNORECASE),
        }

        # Bash patterns
        patterns[CodeLanguage.BASH] = {
            'shebang': re.compile(r'^#!\s*/bin/bash|^#!\s*/bin/sh'),
            'keywords': re.compile(r'\b(if|then|else|elif|fi|for|while|do|done|case|esac|function|return|exit|echo|read|cd|ls|grep|sed|awk)\b'),
            'variables': re.compile(r'\$\{?\w+\}?'),
            'pipes': re.compile(r'\|'),
            'redirections': re.compile(r'[<>]{1,2}'),
        }

        # JSON patterns
        patterns[CodeLanguage.JSON] = {
            'braces': re.compile(r'\{.*\}|\[.*\]'),
            'quotes': re.compile(r'"[^"]*"'),
            'colons': re.compile(r':\s*'),
            'commas': re.compile(r',\s*'),
        }

        # YAML patterns
        patterns[CodeLanguage.YAML] = {
            'key_value': re.compile(r'^\s*\w+\s*:'),
            'lists': re.compile(r'^\s*-\s+'),
            'comments': re.compile(r'#.*$'),
            'quotes': re.compile(r'"[^"]*"|\'[^\']*\''),
        }

        return patterns

    def detect_language(self, code_lines: List[str]) -> Tuple[CodeLanguage, float]:
        """
        Detect the programming language of code lines.

        Args:
            code_lines: List of code lines to analyze

        Returns:
            Tuple of (detected_language, confidence)
        """
        if not code_lines:
            return CodeLanguage.UNKNOWN, 0.0

        code_text = '\n'.join(code_lines)
        scores = {}

        # Score each language
        for language, patterns in self.patterns.items():
            score = self._score_language(code_text, patterns)
            scores[language] = score

        # Find the highest scoring language
        if scores:
            best_language = max(scores, key=scores.get)
            best_score = scores[best_language]

            if best_score >= self.confidence_threshold:
                return best_language, best_score

        return CodeLanguage.UNKNOWN, 0.0

    def _score_language(self, code_text: str, patterns: Dict[str, re.Pattern]) -> float:
        """Score code text against language patterns."""
        score = 0.0

        for pattern_name, pattern in patterns.items():
            matches = len(pattern.findall(code_text))

            # Weight different pattern types differently
            weights = {
                'keywords': 3.0,
                'builtins': 2.0,
                'functions': 2.5,
                'types': 2.0,
                'comments': 1.0,
                'variables': 1.5,
                'includes': 3.0,
                'imports': 2.5,
                'annotations': 2.0,
                'properties': 2.0,
                'linq': 2.0,
                'tags': 4.0,
                'symbols': 2.0,
                'blocks': 1.5,
                'interpolation': 1.5,
                'lifetimes': 2.0,
                'patterns': 2.5,
                'macros': 2.0,
                'clauses': 2.0,
                'braces': 1.0,
                'quotes': 0.5,
                'colons': 0.5,
                'commas': 0.5,
                'key_value': 2.0,
                'lists': 1.5,
                'shebang': 4.0,
                'pipes': 1.0,
                'redirections': 1.0,
                'pointers': 1.5,
                'arrows': 2.0,
                'variables': 2.0,
                'arrays': 1.5,
                'dedent': 1.0,
            }

            weight = weights.get(pattern_name, 1.0)
            score += matches * weight

        # Normalize by code length
        code_length = len(code_text.split())
        if code_length > 0:
            score = score / code_length

        return score


class CodeDetector:
    """Detect code blocks in text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize code detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_lines = self.config.get('min_code_lines', 3)
        self.max_lines = self.config.get('max_code_lines', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

        self.language_detector = LanguageDetector(config)

        # Compile code detection patterns
        self.code_patterns = self._compile_code_patterns()

    def _compile_code_patterns(self) -> List[re.Pattern]:
        """Compile patterns for detecting code blocks."""
        patterns = [
            # Common code indicators
            re.compile(r'^\s*(def|function|class|public|private|protected|if|for|while|do|switch|case)\s*\w', re.MULTILINE),
            re.compile(r'^\s*(import|include|using|from)\s+', re.MULTILINE),
            re.compile(r'^\s*(var|let|const)\s+\w+\s*=', re.MULTILINE),
            re.compile(r'^\s*\w+\s*\([^)]*\)\s*(\{|:)', re.MULTILINE),

            # Language-specific patterns
            re.compile(r'^\s*#\s*include\s*[<"]', re.MULTILINE),  # C/C++
            re.compile(r'^\s*<\?php', re.MULTILINE),  # PHP
            re.compile(r'^\s*#!\s*/bin/(bash|sh)', re.MULTILINE),  # Bash
            re.compile(r'^\s*package\s+\w+', re.MULTILINE),  # Java/Go
            re.compile(r'^\s*func\s+\w+\s*\(', re.MULTILINE),  # Go/Rust
            re.compile(r'^\s*\w+\s*:\s*\w+', re.MULTILINE),  # YAML
            re.compile(r'^\s*-\s+\w+', re.MULTILINE),  # YAML list
            re.compile(r'^\s*[\'"[]\w+[\'"]\s*:', re.MULTILINE),  # JSON key
        ]

        return patterns

    def detect_code_blocks(self, text: str) -> List[CodeBlock]:
        """
        Detect code blocks in text.

        Args:
            text: Text to analyze

        Returns:
            List of detected code blocks
        """
        lines = text.split('\n')
        code_blocks = []

        # Find code block regions
        code_regions = self._find_code_regions(lines)

        # Convert regions to code blocks with language detection
        for start_line, end_line in code_regions:
            block_lines = lines[start_line:end_line + 1]

            # Detect language
            language, confidence = self.language_detector.detect_language(block_lines)

            # Create code block
            code_block = CodeBlock(
                lines=block_lines,
                language=language,
                confidence=confidence,
                start_line=start_line,
                end_line=end_line
            )

            code_blocks.append(code_block)

        return code_blocks

    def _find_code_regions(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Find regions of lines that look like code."""
        regions = []
        current_region = None

        for i, line in enumerate(lines):
            is_code_line = self._is_code_line(line)
            is_empty = not line.strip()

            if is_code_line:
                if current_region is None:
                    # Start a new region
                    current_region = [i, i]
                else:
                    # Extend current region
                    current_region[1] = i
            elif current_region is not None and not is_empty:
                # Non-code, non-empty line ends the current region
                if current_region[1] - current_region[0] + 1 >= self.min_lines:
                    regions.append(tuple(current_region))
                current_region = None

        # Handle region at end of file
        if current_region is not None:
            if current_region[1] - current_region[0] + 1 >= self.min_lines:
                regions.append(tuple(current_region))

        return regions

    def _is_code_line(self, line: str) -> bool:
        """Determine if a line looks like code."""
        line_stripped = line.strip()
        if not line_stripped:
            return False

        # Check against code patterns
        for pattern in self.code_patterns:
            if pattern.search(line):
                return True

        # Check for common code characteristics
        code_indicators = [
            # Contains code-like characters
            any(c in line_stripped for c in '(){}[]<>'),

            # Has programming operators
            any(op in line_stripped for op in ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '&&', '||']),

            # Ends with code-like punctuation
            line_stripped.endswith(('{', '}', ';', ',', '(', ')', '[', ']')),

            # Contains common programming symbols
            any(symbol in line_stripped for symbol in ['->', '=>', '::', '...', '&&', '||']),
        ]

        return any(code_indicators)


class CodeFormatter:
    """Format code blocks with proper markdown syntax highlighting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize code formatter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.include_line_numbers = self.config.get('include_line_numbers', False)
        self.max_code_block_length = self.config.get('max_code_block_length', 1000)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)

    def format_code_block(self, code_block: CodeBlock) -> str:
        """
        Format a code block as markdown.

        Args:
            code_block: Code block to format

        Returns:
            Formatted markdown code block
        """
        if code_block.confidence < self.confidence_threshold:
            # Low confidence - format as plain text with code spans
            return self._format_as_inline_code(code_block.text)

        # High confidence - format as proper code block
        language = code_block.language.value if code_block.language != CodeLanguage.UNKNOWN else ''

        if self.include_line_numbers:
            numbered_lines = []
            for i, line in enumerate(code_block.lines, 1):
                numbered_lines.append(f"{i:3d} | {line}")
            code_text = '\n'.join(numbered_lines)
        else:
            code_text = code_block.text

        # Limit code block length
        if len(code_text) > self.max_code_block_length:
            lines = code_text.split('\n')
            truncated_lines = lines[:50] + ['... (truncated)']
            code_text = '\n'.join(truncated_lines)

        return f"```{language}\n{code_text}\n```"

    def _format_as_inline_code(self, text: str) -> str:
        """Format text as inline code."""
        lines = text.split('\n')
        if len(lines) == 1:
            return f"`{lines[0]}`"
        else:
            return '\n'.join(f"`{line}`" for line in lines)

    def format_text_with_code_blocks(self, text: str) -> str:
        """
        Format text containing code blocks.

        Args:
            text: Text to format

        Returns:
            Text with code blocks properly formatted
        """
        detector = CodeDetector(self.config)
        code_blocks = detector.detect_code_blocks(text)

        if not code_blocks:
            return text

        # Sort code blocks by start line (reverse order to maintain indices)
        code_blocks.sort(key=lambda x: x.start_line, reverse=True)

        # Replace each code block with formatted version
        lines = text.split('\n')
        for block in code_blocks:
            formatted_block = self.format_code_block(block)

            # Replace lines with formatted block
            start_idx = block.start_line
            end_idx = block.end_line

            lines[start_idx:end_idx + 1] = [formatted_block]

        return '\n'.join(lines)


def detect_and_format_code(content: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Detect and format code blocks in content.

    Args:
        content: Content to process
        config: Configuration dictionary

    Returns:
        Content with code blocks formatted as markdown
    """
    formatter = CodeFormatter(config)
    return formatter.format_text_with_code_blocks(content)


def extract_code_blocks(text: str, config: Optional[Dict[str, Any]] = None) -> List[CodeBlock]:
    """
    Extract code blocks from text.

    Args:
        text: Text to analyze
        config: Configuration dictionary

    Returns:
        List of detected code blocks
    """
    detector = CodeDetector(config)
    return detector.detect_code_blocks(text)


def detect_code_language(code: str) -> CodeLanguage:
    """
    Detect the programming language of code.

    Args:
        code: Code string to analyze

    Returns:
        Detected programming language
    """
    detector = LanguageDetector()
    lines = code.split('\n')
    language, _ = detector.detect_language(lines)
    return language
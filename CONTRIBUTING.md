# Contributing to VisionPDF

Thank you for your interest in contributing to VisionPDF! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate
- Use inclusive language
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of PDF processing and vision language models

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/vision-pdf.git
   cd vision-pdf
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/vision-pdf.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Development Dependencies

```bash
# Install in development mode with all optional dependencies
pip install -e .[dev-all]

# Or install specific development dependencies
pip install -e .[dev]
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Verify Installation

```bash
# Test basic functionality
python -c "from vision_pdf import VisionPDF; print('✅ Installation successful')"

# Run tests
pytest

# Check code quality
black --check vision_pdf/ tests/
isort --check-only vision_pdf/ tests/
flake8 vision_pdf/ tests/
mypy vision_pdf/
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vision_pdf --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run specific test file
pytest tests/test_markdown_formatters.py

# Run with verbose output
pytest -v
```

### Test Structure

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests across components
- `tests/fixtures/` - Test data and fixtures
- `conftest.py` - Shared test configuration and fixtures

### Writing Tests

1. **Unit Tests** - Test individual functions and classes
2. **Integration Tests** - Test component interactions
3. **Fixtures** - Use pytest fixtures for setup/teardown
4. **Mocking** - Mock external dependencies (API calls, file I/O)

Example test structure:

```python
import pytest
from vision_pdf.core.processor import VisionPDF

class TestVisionPDF:
    def test_initialization(self):
        converter = VisionPDF()
        assert converter is not None

    @pytest.mark.asyncio
    async def test_pdf_conversion(self, sample_pdf_path):
        converter = VisionPDF()
        result = await converter.convert_pdf(sample_pdf_path)
        assert isinstance(result, str)
        assert len(result) > 0
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.slow          # Slow test (takes > 1 second)
@pytest.mark.requires_ollama  # Requires Ollama backend
```

## Code Quality

### Code Style

We use several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **MyPy** - Type checking

### Formatting Code

```bash
# Format code
black vision_pdf/ tests/
isort vision_pdf/ tests/

# Check formatting
black --check vision_pdf/ tests/
isort --check-only vision_pdf/ tests/

# Lint code
flake8 vision_pdf/ tests/

# Type checking
mypy vision_pdf/
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks:

```bash
# Run all hooks manually
pre-commit run --all-files

# Install hooks (run once)
pre-commit install
```

## Submitting Changes

### 1. Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Ensure all tests pass
- Follow code style guidelines

### 3. Commit Changes

Use descriptive commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(api): add FastAPI REST endpoints
fix(renderer): improve PDF rendering quality
docs(readme): update installation instructions
test(backends): add Ollama integration tests
```

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request on GitHub with:

- Clear title and description
- Reference relevant issues
- Include screenshots for UI changes
- Explain testing performed

## Bug Reports

### Creating Bug Reports

1. **Use the bug report template** in GitHub Issues
2. **Provide detailed information**:
   - VisionPDF version
   - Python version
   - Operating system
   - VLM backend used
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Sample PDF file (if possible)

### Example Bug Report

```markdown
## Bug Description
PDF conversion fails with specific document

## Environment
- VisionPDF: 1.0.0
- Python: 3.9.7
- OS: macOS 12.0
- Backend: Ollama with llava-v1.5

## Steps to Reproduce
1. Run: `vision-pdf convert problem_document.pdf -o output.md`
2. Error occurs after processing page 3

## Expected Behavior
Complete markdown conversion

## Actual Behavior
```
ValueError: Invalid page dimensions detected
```

## Additional Context
- Document contains complex tables
- Error only occurs with hybrid processing mode
```

## Feature Requests

### Proposing Features

1. **Check existing issues** first
2. **Use the feature request template**
3. **Provide clear description** of the feature
4. **Explain the use case** and benefits
5. **Consider implementation** suggestions

### Feature Request Template

```markdown
## Feature Description
Brief description of the proposed feature

## Use Case
Why this feature is needed and how it would be used

## Proposed Solution
How the feature should work

## Alternatives Considered
Other approaches that were considered

## Additional Context
Any relevant information or constraints
```

## Documentation

### Documentation Types

- **API Documentation** - Code docstrings and type hints
- **User Guide** - Usage instructions and examples
- **Development Guide** - Architecture and contribution guidelines
- **Tutorials** - Step-by-step guides

### Writing Documentation

1. **Docstrings** - Use Google-style docstrings:
   ```python
   def convert_pdf(self, pdf_path: str, **kwargs) -> str:
       """Convert PDF to markdown.

       Args:
           pdf_path: Path to the PDF file
           **kwargs: Additional conversion options

       Returns:
           Markdown content as string

       Raises:
           FileNotFoundError: If PDF file doesn't exist
           ConversionError: If conversion fails
       """
   ```

2. **Type Hints** - Include comprehensive type annotations
3. **Examples** - Provide code examples in docstrings
4. **README Updates** - Update README.md for user-facing features

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
cd docs/
make html

# Serve locally
python -m http.server 8000 -d _build/html/
```

## Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** - Breaking changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Release Checklist

1. **Update version numbers** in:
   - `pyproject.toml`
   - `vision_pdf/__init__.py`

2. **Update CHANGELOG.md** with release notes

3. **Run full test suite**:
   ```bash
   pytest --cov=vision_pdf
   ```

4. **Code quality checks**:
   ```bash
   black vision_pdf/ tests/
   isort vision_pdf/ tests/
   flake8 vision_pdf/ tests/
   mypy vision_pdf/
   ```

5. **Build package**:
   ```bash
   python -m build
   ```

6. **Test installation**:
   ```bash
   pip install dist/vision_pdf-*.whl
   ```

7. **Create git tag**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

8. **Deploy to PyPI** (automated via GitHub Actions)

## Getting Help

- **GitHub Issues** - Report bugs and request features
- **GitHub Discussions** - Ask questions and share ideas
- **Documentation** - Check existing docs first
- **Code Review** - Request reviews from maintainers

## Recognizing Contributors

We appreciate all contributions! Contributors will be recognized in:
- README.md contributors section
- Release notes
- Special recognition for significant contributions

Thank you for contributing to VisionPDF! 🚀
# Changelog

All notable changes to VisionPDF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-02

### Added
- **Initial Public Release** - VisionPDF 1.0.0
- **Core PDF Processing** - Advanced PDF to markdown conversion with vision language models
- **Multiple VLM Backends** - Support for Ollama, llama.cpp, and custom API endpoints
- **Processing Modes** - Vision-only, hybrid, and text-only processing modes
- **Format Preservation** - Advanced detection and preservation of tables, math formulas, and code blocks
- **CLI Interface** - Full command-line interface with convert, batch, test, and info commands
- **REST API** - FastAPI-based web API for PDF conversion services
- **Enterprise Features** - Parallel processing, caching, and comprehensive configuration management
- **Documentation** - Complete Sphinx documentation with API reference and tutorials

### Core Features
- **PDF Analysis** - Intelligent document structure detection and layout analysis
- **High-Quality Rendering** - Configurable DPI PDF to image conversion using PyMuPDF
- **Multi-Method Text Extraction** - PyMuPDF, pdfplumber integration with OCR fallback
- **Advanced Markdown Generation** - Layout-aware structure creation with specialized formatters
- **Table Detection & Formatting** - Multiple detection algorithms with professional markdown output
- **Mathematical Expression Recognition** - Pattern-based detection with LaTeX conversion
- **Code Block Identification** - Language detection for 25+ programming languages

### Technical Features
- **Async Processing** - Full async/await support with proper resource management
- **Error Handling** - Comprehensive exception handling with graceful degradation
- **Configuration Management** - Pydantic-based configuration with environment variable support
- **Testing Framework** - Extensive test suite with unit and integration tests (3,131 lines)
- **Code Quality** - Black, isort, MyPy, pytest, and pre-commit hooks configured
- **Multi-Level Caching** - File and memory caching with LRU eviction

### Supported Backends
- **Ollama** - REST API integration with retry logic and model management
- **Llama.cpp** - Python bindings and server API support
- **Custom APIs** - OpenAI and Anthropic compatible API endpoints

### Optional Dependencies
- **OCR Support** - EasyOCR and Tesseract integration for fallback text extraction
- **API Server** - FastAPI and uvicorn for web service deployment
- **Development Tools** - Complete development environment with testing and documentation tools

### CLI Commands
- `convert` - Convert single PDF to markdown
- `batch` - Process multiple PDF files in parallel
- `test` - Test backend connectivity and functionality
- `models` - List available VLM models
- `info` - Display system information and configuration

### API Endpoints
- `GET /health` - System health check
- `POST /convert` - PDF conversion with file upload
- `GET /models` - Available backend models
- `WebSocket /ws/progress` - Real-time conversion progress

### Documentation
- **User Guide** - Complete usage instructions and examples
- **API Reference** - Comprehensive API documentation
- **Development Guide** - Contribution guidelines and architecture overview
- **Tutorials** - Step-by-step guides for common use cases

### Python Support
- **Python 3.8+** - Support for Python 3.8 through 3.12
- **Cross-Platform** - Windows, macOS, and Linux support

### Performance
- **Parallel Processing** - Multi-core PDF processing with resource monitoring
- **Memory Efficient** - Optimized memory usage with cleanup utilities
- **Fast Processing** - High-performance conversion with configurable quality settings

---

## Development Notes

### Architecture
VisionPDF follows a modular architecture with clear separation of concerns:

- `vision_pdf/core/` - Core processing and document models
- `vision_pdf/backends/` - VLM backend implementations
- `vision_pdf/pdf/` - PDF processing (renderer, analyzer, extractor)
- `vision_pdf/markdown/` - Markdown generation with specialized formatters
- `vision_pdf/config/` - Configuration management with Pydantic models
- `vision_pdf/utils/` - Utilities (validation, cache, parallel, images, logging)
- `vision_pdf/cli/` - Command-line interface
- `vision_pdf/api/` - REST API layer

### Quality Assurance
- **Test Coverage** - Comprehensive test suite with unit and integration tests
- **Code Quality** - Automated formatting, linting, and type checking
- **Documentation** - Complete API documentation and user guides
- **CI/CD** - Automated testing and deployment pipeline

### Security
- **Input Validation** - Comprehensive validation and security checks
- **Resource Management** - Proper cleanup and resource monitoring
- **Error Handling** - Secure error handling without information leakage

---

For more detailed information about VisionPDF features and usage, please refer to the [documentation](docs/).
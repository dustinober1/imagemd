# VisionPDF: Convert PDFs to Markdown with Vision Language Models

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/vision-pdf.svg)](https://pypi.org/project/vision-pdf/)

VisionPDF is a Python package that converts PDF documents to well-formatted markdown using state-of-the-art vision language models. It supports multiple VLM backends including Ollama, llama.cpp, and custom API endpoints, making it perfect for both local deployments and enterprise environments.

## ✨ Key Features

- 🧠 **User-configurable Vision Models**: Support for any compatible vision language model
- 🔧 **Multiple Backends**: Ollama, llama.cpp, custom API endpoints, and more
- 🎯 **Processing Flexibility**: Vision-only, hybrid, and text-only processing modes
- 📋 **Full Format Preservation**: Tables, mathematical formulas, code blocks, and layouts
- 🏢 **Enterprise Ready**: Support for internal systems with private LLM deployments
- ⚡ **High Performance**: Parallel processing, intelligent caching, memory management

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install vision-pdf

# With OCR support
pip install vision-pdf[ocr]

# With all optional dependencies
pip install vision-pdf[all]

# For development
pip install vision-pdf[dev-all]
```

### Basic Usage

```python
from vision_pdf import VisionPDF

# Simple conversion with default settings
converter = VisionPDF()
markdown = converter.convert_pdf("document.pdf")
converter.convert_pdf_to_file("document.pdf", "output.md")
```

### Advanced Usage

```python
from vision_pdf import VisionPDF, BackendType, ProcessingMode

# Enterprise deployment with custom settings
converter = VisionPDF(
    backend_type=BackendType.OLLAMA,
    backend_config={
        "model": "llava:7b",
        "base_url": "http://localhost:11434"
    },
    processing_mode=ProcessingMode.HYBRID,
    parallel_processing=True,
    preserve_tables=True,
    preserve_math=True,
    dpi=300
)

markdown = converter.convert_pdf("technical_doc.pdf")
```

### Command Line Interface

```bash
# Basic conversion
vision-pdf convert document.pdf -o output.md

# Advanced processing with specific backend
vision-pdf convert document.pdf \
    --backend ollama \
    --model llava:7b \
    --mode hybrid \
    --preserve-tables \
    --preserve-math \
    --dpi 300 \
    --parallel 4
```

## 📋 Requirements

- Python 3.8+
- A vision language model (Ollama, llama.cpp, or custom API)
- PyMuPDF (for PDF processing)
- Optional: EasyOCR/Tesseract (for OCR support)

## 🏗️ Architecture

VisionPDF follows a modular architecture with separate components for:

- **PDF Processing**: High-quality rendering and layout analysis
- **VLM Backends**: Flexible abstraction for different vision models
- **Markdown Generation**: Layout-aware content formatting
- **Configuration Management**: Hierarchical settings system
- **Performance Optimization**: Parallel processing and caching

## 📖 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [API Documentation](https://vision-pdf.readthedocs.io/)
- [Examples and Tutorials](docs/EXAMPLES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA) - Large Language and Vision Assistant
- [Ollama](https://github.com/ollama/ollama) - Get up and running with Llama 2 and other large language models locally
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - Python bindings for MuPDF's PDF library
- [pdfplumber](https://github.com/jsvine/pdfplumber) - Plumb a PDF for detailed information about each text character
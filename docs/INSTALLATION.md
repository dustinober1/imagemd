# Installation Guide

## System Requirements

*   **Python**: 3.8 or higher
*   **RAM**: 4GB+ recommended
*   **Disk Space**: 1GB+ for models and cache

## Basic Installation

Install VisionPDF using pip:

```bash
pip install vision-pdf
```

For development installation:

```bash
git clone https://github.com/your-repo/visionpdf.git
cd visionpdf
pip install -e .[dev]
```

## Optional Dependencies

VisionPDF has several optional dependencies for enhanced functionality:

```bash
# OCR support
pip install vision-pdf[ocr]

# Performance monitoring
pip install vision-pdf[performance]

# All optional dependencies
pip install vision-pdf[all]
```

## OCR Engines

If you plan to use OCR capabilities (recommended for hybrid processing), you can install specific OCR engines:

```bash
# Tesseract OCR (requires Tesseract binary installed on system)
pip install pytesseract

# EasyOCR (better accuracy but heavier)
pip install easyocr

# PaddleOCR (support for multiple languages)
pip install paddlepaddle paddleocr
```

## Backend Setup

VisionPDF supports multiple backends. You need at least one configured to process documents.

### Ollama Backend

1.  **Install Ollama**:
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

2.  **Pull a model**:
    ```bash
    ollama pull llama2
    # or
    ollama pull llava
    ```

### llama.cpp Backend

1.  **Install llama.cpp**:
    Follow [llama.cpp installation instructions](https://github.com/ggerganov/llama.cpp).

2.  **Download a model**:
    Download a GGUF model compatible with your version.

### Custom API Backend

If you are using a custom inference server (e.g., vLLM, Text Generation Inference, or a cloud provider), no additional local installation is required besides the `vision-pdf` package. See [Configuration](CONFIGURATION.md) for setup details.

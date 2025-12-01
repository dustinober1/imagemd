# Configuration Guide

VisionPDF offers flexible configuration options to tailor the conversion process to your needs.

## Configuration Methods

You can configure VisionPDF using:
1.  **Python Code**: Pass configuration objects directly.
2.  **Environment Variables**: Set environment variables for quick configuration.
3.  **YAML Configuration File**: Load settings from a file.

## Backend Configuration

### Ollama

```python
from vision_pdf import VisionPDF, VisionPDFConfig, BackendType

config = VisionPDFConfig()
config.backends[BackendType.OLLAMA.value].config = {
    "model": "llava:7b",
    "base_url": "http://localhost:11434"
}

processor = VisionPDF(config=config, backend_type=BackendType.OLLAMA)
```

### llama.cpp

```python
config = VisionPDFConfig()
config.backends[BackendType.LLAMA_CPP.value].config = {
    "base_url": "http://localhost:8080"
}

processor = VisionPDF(config=config, backend_type=BackendType.LLAMA_CPP)
```

### Custom API

For air-gapped or custom setups:

```python
config = VisionPDFConfig()
config.backends[BackendType.CUSTOM_API.value].config = {
    "base_url": "https://your-api-endpoint.com",
    "api_key": "your-api-key",
    "model_name": "your-model-name",
    "request_format": "openai",  # or "anthropic", "custom"
    "response_format": "openai"  # or "anthropic", "custom"
}

processor = VisionPDF(config=config, backend_type=BackendType.CUSTOM_API)
```

See [Air-Gapped Environments](AIRGAPPED.md) for more details.

## Processing Options

### Processing Modes

*   **HYBRID** (Default): Combines OCR/text extraction with Vision Model for layout analysis. Best balance of speed and accuracy.
*   **VISION_ONLY**: Uses purely the Vision Model (VLM) to read the document. Best for complex layouts but slower and more token-intensive.
*   **TEXT_ONLY**: Uses standard text extraction. Fastest but loses layout context.

```python
from vision_pdf import ProcessingMode

config.processing.mode = ProcessingMode.HYBRID
```

### Content Preservation

Control what elements are preserved in the output markdown:

```python
config.processing.preserve_tables = True
config.processing.preserve_math = True
config.processing.preserve_code = True
config.processing.preserve_images = False
```

### DPI Settings

Control the resolution for rendering PDF pages to images (for Vision/Hybrid modes):

```python
config.processing.dpi = 300  # Default: 300
```

## OCR Configuration

Configure fallback OCR when VLM confidence is low:

```python
config.processing.ocr_fallback_enabled = True
config.processing.ocr_fallback_threshold = 0.5
config.processing.ocr_config = {
    "engine": "tesseract",  # "tesseract", "easyocr", "paddleocr"
    "languages": ["eng"],
    "confidence_threshold": 0.6,
    "preprocessing": True,
    "deskew": True,
    "enhancement": True
}
```

## Caching

Enable caching to avoid re-processing the same pages:

```python
config.cache.type = "file"  # "memory", "file", "redis", "disabled"
config.cache.directory = "./cache"
config.cache.max_size = 1000
config.cache.ttl = 3600
```

## Performance

```python
config.processing.parallel_processing = True
config.processing.max_workers = 4
config.processing.batch_size = 5
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VISIONPDF_DEFAULT_BACKEND` | Default backend (`ollama`, `llama_cpp`, `custom_api`) |
| `VISIONPDF_PROCESSING_MODE` | Processing mode (`hybrid`, `vision_only`, `text_only`) |
| `VISIONPDF_PROCESSING_DPI` | DPI for rendering (default: 300) |
| `VISIONPDF_OCR_ENABLED` | Enable/disable OCR (`true`, `false`) |
| `VISIONPDF_OCR_ENGINE` | OCR engine to use |
| `VISIONPDF_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, etc.) |

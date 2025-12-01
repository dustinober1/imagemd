# Troubleshooting

## Common Issues

### Backend Connection Failed

**Symptoms**: `BackendConnectionError` or timeouts.

**Solutions**:
1.  **Check Service Status**: Ensure Ollama or your custom server is running.
    ```bash
    curl http://localhost:11434/api/tags  # Ollama
    curl http://localhost:8080/health     # Custom/llama.cpp
    ```
2.  **Verify Configuration**: Check `base_url` and `api_key`.
    ```python
    # Test connection
    connected = await processor.test_backend_connection()
    ```
3.  **Network**: If using Docker, ensure proper network bridging (e.g., `host.docker.internal` on Mac/Windows).

### Low Quality Output

**Symptoms**: Missing text, garbled tables, or hallucinated content.

**Solutions**:
1.  **Change Mode**: Try `ProcessingMode.VISION_ONLY` if layout is complex, or `ProcessingMode.HYBRID` if text is small/dense.
2.  **Increase DPI**: Set `config.processing.dpi = 400` (or higher) for small text.
3.  **Change Model**: Switch to a larger or more capable vision model (e.g., upgrade from 7b to 13b or 34b parameters).
4.  **Enable OCR**: Ensure `ocr_fallback_enabled = True`.

### Memory Errors

**Symptoms**: Process crashes or `OutOfMemoryError`.

**Solutions**:
1.  **Reduce Parallelism**: Set `config.processing.max_workers = 1`.
2.  **Reduce Batch Size**: Set `config.processing.batch_size = 1`.
3.  **Disable Caching**: Or switch to file-based caching if memory-based caching is filling up RAM.

### Missing Dependencies

**Symptoms**: `ImportError` or "Failed to initialize OCR".

**Solutions**:
1.  Install optional dependencies:
    ```bash
    pip install vision-pdf[ocr]
    ```
2.  Install system dependencies (e.g., `tesseract-ocr` package on Linux/Mac).

## Debugging

Enable debug logging to see detailed processing steps:

```python
import logging
import os

# Set via environment variable
os.environ["VISIONPDF_LOG_LEVEL"] = "DEBUG"

# Or configure python logging
logging.basicConfig(level=logging.DEBUG)
```

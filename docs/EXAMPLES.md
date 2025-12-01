# Examples and Tutorials

## Basic Usage

### Simple Conversion

```python
import asyncio
from vision_pdf import VisionPDF

async def simple_convert():
    processor = VisionPDF()
    # Uses default backend (Ollama) and default settings
    markdown = await processor.convert_pdf("document.pdf")
    print(markdown)

if __name__ == "__main__":
    asyncio.run(simple_convert())
```

### Batch Processing

Convert multiple files efficiently:

```python
import asyncio
from vision_pdf import VisionPDF

async def batch_convert():
    processor = VisionPDF()

    files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    output_dir = "./output"

    # Returns list of paths to generated markdown files
    results = await processor.convert_batch(files, output_dir)
    print(f"Converted: {results}")

if __name__ == "__main__":
    asyncio.run(batch_convert())
```

## Specific Use Cases

### Academic Papers

Preserve mathematical formulas and tables:

```python
from vision_pdf import VisionPDF, VisionPDFConfig, ProcessingMode

config = VisionPDFConfig()
config.processing.preserve_math = True
config.processing.preserve_tables = True
config.processing.mode = ProcessingMode.HYBRID

# Use a model known for good reasoning/math support
config.backends["ollama"].config["model"] = "llava:13b"

processor = VisionPDF(config=config)
# ... use processor
```

### Technical Documentation

Preserve code blocks and technical formatting:

```python
config = VisionPDFConfig()
config.processing.preserve_code = True
config.processing.mode = ProcessingMode.HYBRID

processor = VisionPDF(config=config)
```

### Scanned Documents

Process older scanned documents using robust OCR settings:

```python
config = VisionPDFConfig()
config.processing.mode = ProcessingMode.VISION_ONLY  # Rely on vision model for layout
config.processing.ocr_fallback_enabled = True
config.processing.ocr_config = {
    "engine": "tesseract",
    "preprocessing": True,
    "enhancement": True,
    "deskew": True
}

processor = VisionPDF(config=config)
```

### Air-Gapped / Enterprise

See [Air-Gapped Environments](AIRGAPPED.md) for detailed examples on connecting to internal API endpoints.

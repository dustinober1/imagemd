User Guide
==========

This comprehensive guide covers how to use VisionPDF effectively for various PDF to markdown conversion tasks.

Table of Contents
----------------

.. contents::
   :local:
   :depth: 2

Installation
------------

System Requirements
~~~~~~~~~~~~~~~~~~

* Python 3.8 or higher
* 4GB+ RAM recommended
* 1GB+ disk space for models and cache

Basic Installation
~~~~~~~~~~~~~~~~~~

Install VisionPDF using pip:

.. code-block:: bash

   pip install visionpdf

For development installation:

.. code-block:: bash

   git clone https://github.com/your-repo/visionpdf.git
   cd visionpdf
   pip install -e .[dev]

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

VisionPDF has several optional dependencies for enhanced functionality:

.. code-block:: bash

   # OCR support
   pip install visionpdf[ocr]

   # Performance monitoring
   pip install visionpdf[performance]

   # All optional dependencies
   pip install visionpdf[all]

Or install specific OCR engines:

.. code-block:: bash

   # Tesseract OCR
   pip install pytesseract

   # EasyOCR
   pip install easyocr

   # PaddleOCR
   pip install paddlepaddle paddleocr

Backend Setup
--------------

Ollama Backend
~~~~~~~~~~~~~~~

1. Install Ollama:

.. code-block:: bash

   curl -fsSL https://ollama.ai/install.sh | sh

2. Pull a model:

.. code-block:: bash

   ollama pull llama2
   ollama pull codellama

3. Configure VisionPDF:

.. code-block:: python

   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType

   config = VisionPDFConfig()
   config.backends[BackendType.OLLAMA.value].config = {
       "model": "llama2",
       "host": "http://localhost:11434"
   }

   processor = VisionPDF(config=config, backend_type=BackendType.OLLAMA)

llama.cpp Backend
~~~~~~~~~~~~~~~~~

1. Install llama.cpp:

.. code-block:: bash

   # Follow llama.cpp installation instructions
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make

2. Download a model and convert it:

.. code-block:: bash

   ./main -m models/llama-2-7b.ggmlv3.q4_0.bin --color -n -1

3. Configure VisionPDF:

.. code-block:: python

   config = VisionPDFConfig()
   config.backends[BackendType.LLAMA_CPP.value].config = {
       "model_path": "/path/to/model.ggml",
       "host": "http://localhost:8080"
   }

   processor = VisionPDF(config=config, backend_type=BackendType.LLAMA_CPP)

Custom API Backend
~~~~~~~~~~~~~~~~~~

For custom API backends, configure the endpoint and authentication:

.. code-block:: python

   config = VisionPDFConfig()
   config.backends[BackendType.CUSTOM_API.value].config = {
       "api_url": "https://your-api-endpoint.com/v1/chat/completions",
       "api_key": "your-api-key",
       "model": "your-model-name"
   }

   processor = VisionPDF(config=config, backend_type=BackendType.CUSTOM_API)

Basic Usage
-----------

Simple Conversion
~~~~~~~~~~~~~~~~

The most basic usage is converting a PDF to markdown:

.. code-block:: python

   import asyncio
   from vision_pdf import VisionPDF

   async def simple_convert():
       processor = VisionPDF()

       # Convert to string
       markdown = await processor.convert_pdf("document.pdf")
       print(markdown)

   asyncio.run(simple_convert())

File Output
~~~~~~~~~~

Convert directly to a file:

.. code-block:: python

   async def convert_to_file():
       processor = VisionPDF()
       await processor.convert_pdf_to_file(
           "input.pdf",
           "output.md"
       )

   asyncio.run(convert_to_file())

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple PDF files:

.. code-block:: python

   async def batch_convert():
       processor = VisionPDF()

       pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
       output_dir = "converted_docs"

       results = await processor.convert_batch(pdf_files, output_dir)

       print(f"Converted {len(results)} files:")
       for result in results:
           print(f"  - {result}")

   asyncio.run(batch_convert())

Progress Monitoring
~~~~~~~~~~~~~~~~~~~

Track progress of batch processing:

.. code-block:: python

   def progress_callback(current, total, filename):
       percent = (current / total) * 100
       print(f"Progress: {percent:.1f}% - {filename}")

   async def convert_with_progress():
       processor = VisionPDF()

       pdf_files = ["large_doc1.pdf", "large_doc2.pdf"]
       output_dir = "output"

       await processor.convert_batch(
           pdf_files,
           output_dir,
           progress_callback=progress_callback
       )

   asyncio.run(convert_with_progress())

Advanced Configuration
-----------------------

Processing Modes
~~~~~~~~~~~~~~~~

VisionPDF supports three processing modes:

Text-Only Mode
~~~~~~~~~~~~~~

Extracts text without vision processing:

.. code-block:: python

   from vision_pdf import VisionPDF, VisionPDFConfig, ProcessingMode

   config = VisionPDFConfig()
   config.processing.mode = ProcessingMode.TEXT_ONLY

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("document.pdf")

Vision Mode
~~~~~~~~~~

Uses only vision language models:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.mode = ProcessingMode.VISION

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("document.pdf")

Hybrid Mode (Default)
~~~~~~~~~~~~~~~~~~~~

Combines text extraction with vision processing:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.mode = ProcessingMode.HYBRID

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("document.pdf")

Content Preservation
~~~~~~~~~~~~~~~~~~~~

Configure which content types to preserve:

.. code-block:: python

   config = VisionPDFConfig()

   # Preserve tables
   config.processing.preserve_tables = True

   # Preserve mathematical expressions
   config.processing.preserve_math = True

   # Preserve code blocks
   config.processing.preserve_code = True

   # Preserve images (as references)
   config.processing.preserve_images = False

   processor = VisionPDF(config=config)

OCR Fallback
~~~~~~~~~~~~

Configure OCR fallback when VLM processing fails:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.ocr_fallback_enabled = True
   config.processing.ocr_fallback_threshold = 0.5  # Confidence threshold
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["eng", "spa", "fra"],
       "confidence_threshold": 0.6,
       "preprocessing": True,
       "deskew": True,
       "enhancement": True
   }

   processor = VisionPDF(config=config)

Caching Configuration
~~~~~~~~~~~~~~~~~~~~

Configure result caching for performance:

.. code-block:: python

   config = VisionPDFConfig()
   config.cache.enabled = True
   config.cache.type = "file"  # "memory", "file", or "redis"
   config.cache.directory = "/path/to/cache"
   config.cache.max_size_mb = 1024
   config.cache.ttl_seconds = 3600

   processor = VisionPDF(config=config)

Performance Optimization
-----------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Configure worker count for parallel processing:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.parallel_processing = True
   config.processing.max_workers = 8
   config.processing.batch_size = 5

   processor = VisionPDF(config=config)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~

Enable performance monitoring:

.. code-block:: python

   from vision_pdf.utils.performance import PerformanceOptimizer, measure_performance

   config = {
       'monitoring_enabled': True,
       'cache_enabled': True,
       'parallel_processing': True,
       'max_workers': 8,
       'batch_size': 10
   }

   optimizer = PerformanceOptimizer(config)

   # Monitor specific operations
   @measure_performance("pdf_conversion")
   async def convert_pdf_monitored(pdf_path):
       processor = VisionPDF(config=optimizer.config)
       return await processor.convert_pdf(pdf_path)

   # Generate performance report
   report = optimizer.generate_performance_report()
   print(f"Performance metrics: {report}")

Memory Management
~~~~~~~~~~~~~~~~

Configure memory limits and cleanup:

.. code-block:: python

   config = VisionPDFConfig()

   # Limit memory usage
   config.cache.max_memory_entries = 1000
   config.cache.max_size_mb = 512

   # Enable automatic cleanup
   config.cache.auto_cleanup = True
   config.cache.cleanup_interval = 3600

   processor = VisionPDF(config=config)

Specific Use Cases
-----------------

Academic Papers
~~~~~~~~~~~~~~~

Process academic papers with mathematical formulas:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.preserve_math = True
   config.processing.preserve_tables = True
   config.processing.mode = ProcessingMode.HYBRID
   config.processing.ocr_fallback_enabled = True

   # Use a model good at academic content
   config.backends[BackendType.OLLAMA.value].config = {
       "model": "codellama"  # Good for academic content
   }

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("research_paper.pdf")

Technical Documentation
~~~~~~~~~~~~~~~~~~~~

Process API documentation and code examples:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.preserve_code = True
   config.processing.preserve_tables = True
   config.processing.mode = ProcessingMode.HYBRID

   # Use code-focused model
   config.backends[BackendType.OLLAMA.value].config = {
       "model": "codellama:7b-code"
   }

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("api_docs.pdf")

Financial Documents
~~~~~~~~~~~~~~~~~~

Process financial reports with tables and numbers:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.preserve_tables = True
   config.processing.mode = ProcessingMode.VISION
   config.processing.ocr_fallback_enabled = True

   # Configure OCR for financial documents
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["eng"],
       "preprocessing": True,
       "enhancement": True
   }

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("financial_report.pdf")

Scanned Documents
~~~~~~~~~~~~~~~~~

Process scanned PDFs requiring OCR:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.ocr_fallback_enabled = True
   config.processing.ocr_fallback_threshold = 0.3  # Lower threshold for scanned docs
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["eng"],
       "preprocessing": True,
       "deskew": True,
       "enhancement": True
   }

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("scanned_document.pdf")

Error Handling
-------------

Common Exceptions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.utils.exceptions import (
       VisionPDFError,
       PDFProcessingError,
       BackendError,
       ValidationError,
       ConfigurationError
   )

   async def safe_convert():
       try:
           processor = VisionPDF()
           result = await processor.convert_pdf("document.pdf")
           return result

       except ValidationError as e:
           print(f"Invalid input: {e}")
           # Check file path, format, permissions

       except BackendError as e:
           print(f"Backend error: {e}")
           # Check backend configuration, connection

       except PDFProcessingError as e:
           print(f"PDF processing error: {e}")
           # Check PDF file integrity

       except VisionPDFError as e:
           print(f"VisionPDF error: {e}")
           # General error handling

   asyncio.run(safe_convert())

Retry Logic
~~~~~~~~~~

Implement retry logic for robust processing:

.. code-block:: python

   import asyncio
   from vision_pdf.utils.exceptions import BackendError

   async def convert_with_retry(pdf_path, max_retries=3):
       for attempt in range(max_retries):
           try:
               processor = VisionPDF()
               return await processor.convert_pdf(pdf_path)

           except BackendError as e:
               if attempt == max_retries - 1:
                   raise

               print(f"Backend error on attempt {attempt + 1}, retrying...")
               await asyncio.sleep(2 ** attempt)  # Exponential backoff

   # Usage
   result = await convert_with_retry("problematic.pdf")

Custom Prompts
-------------

Modify Processing Prompts
~~~~~~~~~~~~~~~~~~~~~~~~

Customize the prompts sent to VLMs:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.custom_prompt = (
       "Convert this PDF page to high-quality markdown. "
       "Pay special attention to: "
       "1. Preserve exact table structures "
       "2. Convert all mathematical expressions to LaTeX "
       "3. Identify and format code blocks with proper syntax highlighting "
       "4. Maintain the original document hierarchy and formatting "
       "5. Handle multi-column layouts correctly"
   )

   processor = VisionPDF(config=config)

Language-Specific Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process documents in different languages:

.. code-block:: python

   # Spanish documents
   config = VisionPDFConfig()
   config.processing.language = "es"
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["spa", "eng"],
       "confidence_threshold": 0.6
   }

   # French documents
   config.processing.language = "fr"
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["fra", "eng"],
       "confidence_threshold": 0.6
   }

CLI Usage
---------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~

VisionPDF provides a command-line interface:

.. code-block:: bash

   # Basic conversion
   visionpdf convert input.pdf output.md

   # Specify backend
   visionpdf convert --backend ollama --model llama2 input.pdf output.md

   # Batch processing
   visionpdf batch --input-dir ./pdfs --output-dir ./markdown

   # With configuration
   visionpdf convert --config config.yaml input.pdf output.md

Configuration File
~~~~~~~~~~~~~~~~~~

Use a YAML configuration file:

.. code-block:: yaml

   # config.yaml
   backend:
     type: "ollama"
     config:
       model: "llama2"
       host: "http://localhost:11434"

   processing:
     mode: "hybrid"
     preserve_tables: true
     preserve_math: true
     preserve_code: true
     ocr_fallback_enabled: true

   cache:
     enabled: true
     type: "file"
     directory: "./cache"
     max_size_mb: 1024

   performance:
     parallel_processing: true
     max_workers: 4

.. code-block:: bash

   visionpdf convert --config config.yaml input.pdf output.md

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~~

Backend Connection Failed
~~~~~~~~~~~~~~~~~~~~~~~~

1. Check if the backend is running:

.. code-block:: bash

   # For Ollama
   ollama list

   # For llama.cpp
   curl http://localhost:8080/health

2. Verify configuration:

.. code-block:: python

   processor = VisionPDF()
   connection_ok = await processor.test_backend_connection()
   print(f"Backend connection: {'OK' if connection_ok else 'FAILED'})

Memory Issues
~~~~~~~~~~~~

1. Reduce batch size and worker count:

.. code-block:: python

   config = VisionPDFConfig()
   config.processing.max_workers = 2  # Reduce from default
   config.processing.batch_size = 3   # Reduce from default
   config.cache.max_memory_entries = 100  # Reduce cache

2. Enable memory monitoring:

.. code-block:: python

   from vision_pdf.utils.performance import ResourceMonitor

   monitor = ResourceMonitor()
   with monitor.monitor_memory("conversion"):
       result = await processor.convert_pdf("large_file.pdf")

OCR Quality Issues
~~~~~~~~~~~~~~~~~~

1. Try different OCR engines:

.. code-block:: python

   # Test Tesseract
   config.processing.ocr_config = {"engine": "tesseract"}

   # Test EasyOCR
   config.processing.ocr_config = {"engine": "easyocr"}

   # Test PaddleOCR
   config.processing.ocr_config = {"engine": "paddleocr"}

2. Improve preprocessing:

.. code-block:: python

   config.processing.ocr_config = {
       "preprocessing": True,
       "deskew": True,
       "enhancement": True,
       "confidence_threshold": 0.7  # Increase threshold
   }

Debug Mode
~~~~~~~~~~

Enable debug logging:

.. code-block:: python

   import logging
   from vision_pdf.utils.logging_config import setup_logging

   setup_logging(level=logging.DEBUG)

   # Or set environment variable
   # export VISIONPDF_LOG_LEVEL=DEBUG

Performance Tips
----------------

Large Documents
~~~~~~~~~~~~~~

1. Use batch processing
2. Enable caching
3. Increase memory allocation
4. Use SSD for cache storage

.. code-block:: python

   config = VisionPDFConfig()
   config.cache.enabled = True
   config.cache.max_size_mb = 2048  # Increase cache size
   config.processing.max_workers = 8   # Increase parallelism

High-Volume Processing
~~~~~~~~~~~~~~~~~~~~~

1. Use file-based caching with SSD
2. Optimize batch sizes
3. Monitor resource usage
4. Implement retry logic

.. code-block:: python

   config = VisionPDFConfig()
   config.cache.type = "file"
   config.cache.directory = "/fast-ssd/cache"
   config.processing.batch_size = 10
   config.processing.max_workers = 6

Network Latency
~~~~~~~~~~~~~~~~

For custom API backends with high latency:

1. Increase timeouts
2. Use connection pooling
3. Implement request batching

.. code-block:: python

   config = VisionPDFConfig()
   config.backends[BackendType.CUSTOM_API.value].config = {
       "timeout": 120,  # Increase timeout
       "max_retries": 3,
       "retry_delay": 2.0
   }
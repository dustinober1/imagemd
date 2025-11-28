API Reference
=============

This section provides detailed documentation for all VisionPDF classes, methods, and functions.

Core API
--------

VisionPDF Processor
~~~~~~~~~~~~~~~~~~~

.. autoclass:: vision_pdf.VisionPDF
   :members:
   :inherited-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. autoclass:: vision_pdf.config.settings.VisionPDFConfig
   :members:
   :show-inheritance:

.. autoclass:: vision_pdf.config.settings.ProcessingConfig
   :members:
   :show-inheritance:

.. autoclass:: vision_pdf.config.settings.BackendConfig
   :members:
   :show-inheritance:

Backends
--------

Base Backend
~~~~~~~~~~~~

.. automodule:: vision_pdf.backends.base
   :members:
   :show-inheritance:

Ollama Backend
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.backends.ollama
   :members:
   :show-inheritance:

llama.cpp Backend
~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.backends.llama_cpp
   :members:
   :show-inheritance:

Custom API Backend
~~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.backends.custom
   :members:
   :show-inheritance:

Document Model
--------------

Document Classes
~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.core.document
   :members:
   :show-inheritance:

PDF Processing
~~~~~~~~~~~~

.. automodule:: vision_pdf.core.processor
   :members:
   :show-inheritance:

PDF Analysis
~~~~~~~~~~~~

.. automodule:: vision_pdf.pdf.analyzer
   :members:
   :show-inheritance:

PDF Rendering
~~~~~~~~~~~~

.. automodule:: vision_pdf.pdf.renderer
   :members:
   :show-inheritance:

Text Extraction
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.pdf.extractor
   :members:
   :show-inheritance:

Markdown Generation
-------------------

Main Generator
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.generator
   :members:
   :show-inheritance:

Advanced Formatters
~~~~~~~~~~~~~~~~~~

Table Formatter
~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.formatters.tables
   :members:
   :show-inheritance:

Math Formatter
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.formatters.math
   :members:
   :show-inheritance:

Code Formatter
~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.formatters.code
   :members:
   :show-inheritance:

OCR Integration
---------------

OCR Base Classes
~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.ocr.base
   :members:
   :show-inheritance:

Tesseract Engine
~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.ocr.engines.tesseract_engine
   :members:
   :show-inheritance:

EasyOCR Engine
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.ocr.engines.easyocr_engine
   :members:
   :show-inheritance:

PaddleOCR Engine
~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.ocr.engines.paddleocr_engine
   :members:
   :show-inheritance:

Performance and Caching
------------------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.utils.performance
   :members:
   :show-inheritance:

Caching System
~~~~~~~~~~~~~

.. automodule:: vision_pdf.utils.cache
   :members:
   :show-inheritance:

Logging and Utilities
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.utils.logging_config
   :members:
   :show-inheritance:

.. automodule:: vision_pdf.utils.exceptions
   :members:
   :show-inheritance:

CLI Interface
-------------

Command Line Tool
~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.cli.main
   :members:
   :show-inheritance:

Examples
--------

Usage Examples
~~~~~~~~~~~~

Here are some practical examples of how to use VisionPDF:

Basic Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf import VisionPDF

   # Simple conversion
   processor = VisionPDF()
   await processor.convert_pdf_to_file("input.pdf", "output.md")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode

   config = VisionPDFConfig()
   config.default_backend = BackendType.OLLAMA
   config.processing.mode = ProcessingMode.HYBRID
   config.processing.preserve_tables = True
   config.processing.preserve_math = True
   config.processing.preserve_code = True

   processor = VisionPDF(config=config)
   result = await processor.convert_pdf("complex_document.pdf")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple PDFs
   results = await processor.convert_batch(
       ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
       "output_directory"
   )

   # Process with progress callback
   def progress_callback(current, total, filename):
       print(f"Progress: {current}/{total} - {filename}")

   results = await processor.convert_batch(
       pdf_files,
       output_dir,
       progress_callback=progress_callback
   )

Custom Backend
~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.backends.base import VLMBackend, ProcessingRequest, ProcessingResponse

   class MyCustomBackend(VLMBackend):
       async def initialize(self):
           # Initialize your custom VLM
           pass

       async def process_page(self, request: ProcessingRequest) -> ProcessingResponse:
           # Process with your VLM
           response = await my_vlm.process(request.image_path)

           return ProcessingResponse(
               markdown=response.text,
               confidence=response.confidence,
               processing_time=response.processing_time
           )

   # Use custom backend
   processor = VisionPDF(backend_type=BackendType.CUSTOM_API)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.utils.performance import PerformanceOptimizer, measure_performance

   config = {
       'monitoring_enabled': True,
       'cache_enabled': True,
       'parallel_processing': True,
       'max_workers': 8
   }

   optimizer = PerformanceOptimizer(config)

   # Monitor performance
   with optimizer.measure_operation("pdf_conversion"):
       result = await processor.convert_pdf("large_document.pdf")

   # Get performance report
   report = optimizer.generate_performance_report("performance_report.json")

Error Handling
~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.utils.exceptions import (
       VisionPDFError, PDFProcessingError, BackendError
   )

   try:
       processor = VisionPDF()
       result = await processor.convert_pdf("document.pdf")
   except BackendError as e:
       print(f"Backend error: {e}")
       # Try fallback or different backend
   except PDFProcessingError as e:
       print(f"PDF processing error: {e}")
       # Check PDF file or permissions
   except VisionPDFError as e:
       print(f"General VisionPDF error: {e}")
       # Handle other errors

Type Hints
----------

VisionPDF uses comprehensive type hints throughout the codebase. Here are the main types:

.. code-block:: python

   from typing import Optional, List, Dict, Any, Union, Callable
   from pathlib import Path

   # Core types
   PDFPath = Union[str, Path]
   OutputPath = Union[str, Path]
   ProgressCallback = Callable[[int, int, str], None]

   # Backend types
   BackendConfig = Dict[str, Any]
   ProcessingMode = str  # "text_only", "vision", "hybrid"

   # Content types
   ContentType = str  # "text", "table", "math", "code", "image"

Constants
---------

.. code-block:: python

   # Backend types
   from vision_pdf.config.settings import BackendType
   BackendType.OLLAMA      # Ollama local models
   BackendType.LLAMA_CPP   # llama.cpp models
   BackendType.CUSTOM_API  # Custom API backends

   # Processing modes
   from vision_pdf.config.settings import ProcessingMode
   ProcessingMode.TEXT_ONLY  # Text extraction only
   ProcessingMode.VISION     # Vision model processing
   ProcessingMode.HYBRID     # Combined approach

   # Content types
   from vision_pdf.core.document import ContentType
   ContentType.TEXT         # Regular text content
   ContentType.TABLE        # Tabular data
   ContentType.MATHEMATICAL # Math expressions
   ContentType.CODE         # Code blocks
   ContentType.IMAGE        # Images
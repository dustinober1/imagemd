VisionPDF Documentation
========================

VisionPDF is a powerful Python package that converts PDF documents to markdown using vision language models. It provides advanced content recognition, including tables, mathematical expressions, and code blocks, with support for multiple VLM backends.

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: https://github.com/your-repo/visionpdf
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.8+-green.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-orange.svg
   :target: LICENSE
   :alt: License

Features
--------

* **Vision-Based Processing**: Leverage state-of-the-art vision language models for accurate PDF interpretation
* **Multiple VLM Backends**: Support for Ollama, llama.cpp, and custom API backends
* **Advanced Content Recognition**: Automatic detection and formatting of tables, mathematical expressions, and code blocks
* **OCR Fallback**: Robust OCR integration when VLM processing fails or produces low confidence results
* **Performance Optimization**: Intelligent caching and batch processing for large documents
* **Flexible Configuration**: Extensive customization options for different use cases

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install VisionPDF using pip:

.. code-block:: bash

   pip install visionpdf

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from vision_pdf import VisionPDF

   async def convert_pdf():
       processor = VisionPDF()

       # Convert a single PDF
       markdown = await processor.convert_pdf("document.pdf")
       print(markdown)

       # Convert to file
       await processor.convert_pdf_to_file("document.pdf", "output.md")

       # Batch process multiple PDFs
       results = await processor.convert_batch(
           ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
           "output_directory"
       )
       print(f"Processed {len(results)} files")

   # Run the async function
   asyncio.run(convert_pdf())

Configuration
~~~~~~~~~~~~~

VisionPDF is highly configurable:

.. code-block:: python

   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode

   config = VisionPDFConfig()

   # Configure backend
   config.default_backend = BackendType.OLLAMA

   # Configure processing
   config.processing.mode = ProcessingMode.HYBRID
   config.processing.preserve_tables = True
   config.processing.preserve_math = True
   config.processing.preserve_code = True

   # Configure OCR fallback
   config.processing.ocr_fallback_enabled = True
   config.processing.ocr_config = {
       "engine": "tesseract",
       "languages": ["eng", "spa"]
   }

   processor = VisionPDF(config=config)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   backends
   configuration
   performance
   contributing
   changelog

API Reference
-------------

Core Classes
~~~~~~~~~~~~

.. autoclass:: vision_pdf.VisionPDF
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: vision_pdf.config.settings.VisionPDFConfig
   :members:
   :show-inheritance:

Backends
~~~~~~~~

.. automodule:: vision_pdf.backends.base
   :members:
   :show-inheritance:

.. automodule:: vision_pdf.backends.ollama
   :members:
   :show-inheritance:

Processors
~~~~~~~~~~

.. automodule:: vision_pdf.core.processor
   :members:
   :show-inheritance:

Document Model
~~~~~~~~~~~~~~

.. automodule:: vision_pdf.core.document
   :members:
   :show-inheritance:

Markdown Generation
~~~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.generator
   :members:
   :show-inheritance:

Advanced Formatters
~~~~~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.markdown.formatters.tables
   :members:
   :show-inheritance:

.. automodule:: vision_pdf.markdown.formatters.math
   :members:
   :show-inheritance:

.. automodule:: vision_pdf.markdown.formatters.code
   :members:
   :show-inheritance:

OCR Integration
~~~~~~~~~~~~~~~

.. automodule:: vision_pdf.ocr.base
   :members:
   :show-inheritance:

Performance
~~~~~~~~~~

.. automodule:: vision_pdf.utils.performance
   :members:
   :show-inheritance:

.. automodule:: vision_pdf.utils.cache
   :members:
   :show-inheritance:

Examples
--------

Converting Different Types of PDFs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Academic paper with mathematical formulas
   async def convert_academic_paper():
       config = VisionPDFConfig()
       config.processing.preserve_math = True
       config.processing.preserve_tables = True

       processor = VisionPDF(config=config)
       markdown = await processor.convert_pdf("research_paper.pdf")
       return markdown

   # Code documentation
   async def convert_code_docs():
       config = VisionPDFConfig()
       config.processing.preserve_code = True
       config.processing.ocr_fallback_enabled = True

       processor = VisionPDF(config=config)
       markdown = await processor.convert_pdf("api_documentation.pdf")
       return markdown

Custom Backend Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.backends.base import VLMBackend
   from vision_pdf import VisionPDF, VisionPDFConfig

   class CustomVLMBackend(VLMBackend):
       async def initialize(self):
           # Initialize your custom VLM
           pass

       async def process_page(self, request):
           # Process page with your VLM
           response = await your_vlm.process(request.image_path)
           return ProcessingResponse(
               markdown=response.text,
               confidence=response.confidence
           )

   # Use custom backend
   processor = VisionPDF(backend_type=BackendType.CUSTOM_API)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf.utils.performance import PerformanceOptimizer

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
   report = optimizer.generate_performance_report()
   print(f"Performance metrics: {report}")

Contributing
------------

We welcome contributions! Please see our `Contributing Guide`_ for details.

.. _Contributing Guide: contributing.html

License
-------

VisionPDF is licensed under the MIT License. See the `LICENSE`_ file for details.

.. _LICENSE: https://github.com/your-repo/visionpdf/blob/main/LICENSE

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
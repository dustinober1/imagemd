VisionPDF Documentation
======================

VisionPDF is a Python package that converts PDF documents to well-formatted markdown using vision language models. It supports multiple VLM backends including Ollama, llama.cpp, and custom API endpoints, making it perfect for both local deployments and enterprise environments.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://python.org
.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE
.. image:: https://img.shields.io/pypi/v/vision-pdf.svg
   :target: https://pypi.org/project/vision-pdf/

Key Features
------------

- 🧠 **User-configurable Vision Models**: Support for any compatible vision language model
- 🔧 **Multiple Backends**: Ollama, llama.cpp, custom API endpoints, and more
- 🎯 **Processing Flexibility**: Vision-only, hybrid, and text-only processing modes
- 📋 **Full Format Preservation**: Tables, mathematical formulas, code blocks, and layouts
- 🏢 **Enterprise Ready**: Support for internal systems with private LLM deployments
- ⚡ **High Performance**: Parallel processing, intelligent caching, memory management

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Basic installation
   pip install vision-pdf

   # With OCR support
   pip install vision-pdf[ocr]

   # With all optional dependencies
   pip install vision-pdf[all]

Basic Usage
~~~~~~~~~~~~~

.. code-block:: python

   from vision_pdf import VisionPDF

   # Simple conversion with default settings
   converter = VisionPDF()
   markdown = converter.convert_pdf("document.pdf")
   converter.convert_pdf_to_file("document.pdf", "output.md")

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

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

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

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

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   configuration
   quickstart
   examples
   troubleshooting

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   core
   pdf
   backends
   markdown
   config
   utils

Development
-----------

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   architecture
   changelog

Tutorials
---------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_usage
   tutorials/backend_configuration
   tutorials/enterprise_deployment
   tutorials/performance_optimization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
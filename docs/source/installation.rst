Installation Guide
==================

This guide will help you install VisionPDF and its dependencies.

System Requirements
-------------------

* Python 3.8 or higher
* A vision language model (Ollama, llama.cpp, or custom API)
* 4GB+ RAM recommended for most operations
* Disk space for temporary files and caching

Basic Installation
------------------

Using pip (recommended)::

   pip install vision-pdf

This installs the core package with basic PDF processing capabilities.

Optional Dependencies
---------------------

OCR Support
~~~~~~~~~~~

Install with OCR support for better text extraction from scanned PDFs::

   pip install vision-pdf[ocr]

This includes:
* EasyOCR - General-purpose OCR with high accuracy
* Tesseract - Fast OCR engine for structured documents

Vision Model Backends
~~~~~~~~~~~~~~~~~~~~~

Ollama Backend
~~~~~~~~~~~~~~~

For local Ollama deployment::

   pip install vision-pdf[ollama]

llama.cpp Backend
~~~~~~~~~~~~~~~~

For llama.cpp server integration::

   pip install vision-pdf[llama-cpp]

All Dependencies
~~~~~~~~~~~~~~~~

Install all optional dependencies at once::

   pip install vision-pdf[all]

Development Installation
------------------------

For developers who want to contribute to VisionPDF::

   # Clone the repository
   git clone https://github.com/visionpdf/vision-pdf.git
   cd vision-pdf

   # Install in development mode with all dependencies
   pip install -e .[dev-all]

   # Install pre-commit hooks
   pre-commit install

Backend Setup
-------------

Ollama Setup
~~~~~~~~~~~~

1. Install Ollama: https://ollama.ai/download

2. Pull a vision model::

   ollama pull llava
   # Or for better performance
   ollama pull llava:7b

3. Start Ollama server::

   ollama serve

4. Verify installation::

   ollama list

llama.cpp Setup
~~~~~~~~~~~~~~~

1. Install llama.cpp: https://github.com/ggerganov/llama.cpp

2. Download a vision model in GGUF format

3. Start llama.cpp server::

   ./main -m model.gguf --host 0.0.0.0 --port 8080 --ctx-size 2048

Custom API Setup
~~~~~~~~~~~~~~~~

For custom API endpoints, ensure you have:

* API endpoint URL
* Authentication credentials (if required)
* Compatible vision model

Docker Installation
-------------------

We provide Docker images for easy deployment:

Using pre-built image::

   docker pull visionpdf/vision-pdf:latest

Building from source::

   docker build -t vision-pdf .

Running with Docker::

   docker run -v $(pwd)/input:/app/input \
              -v $(pwd)/output:/app/output \
              visionpdf convert input/document.pdf -o output/document.md

Verification
------------

To verify your installation:

1. Test basic import::

   python -c "import vision_pdf; print('Installation successful!')"

2. Test with a sample PDF::

   from vision_pdf import VisionPDF
   converter = VisionPDF()
   print("VisionPDF ready to use!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: Ensure you're using Python 3.8+ and installed dependencies correctly.

**PyMuPDF Installation**: If PyMuPDF fails to install, try::

   pip install --upgrade pip
   pip install pymupdf

**OCR Dependencies**: For OCR support, you may need system packages:

Ubuntu/Debian::

   sudo apt-get update
   sudo apt-get install tesseract-ocr libtesseract-dev

macOS::

   brew install tesseract

**Backend Connection**: Ensure your VLM backend is running and accessible at the configured URL.

Performance Tips
---------------

* Use SSD storage for better I/O performance
* Allocate sufficient RAM for your models (8GB+ recommended for 7B models)
* Use parallel processing for batch operations
* Enable caching for repeated conversions

Next Steps
----------

* Read the :doc:`configuration` guide to customize your setup
* Try the :doc:`quickstart` tutorial
* Explore :doc:`examples` for advanced usage patterns
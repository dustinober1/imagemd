Tutorials
=========

This section provides step-by-step tutorials for common VisionPDF use cases.

Table of Contents
----------------

.. contents::
   :local:
   :depth: 2

Tutorial 1: Getting Started
---------------------------

This tutorial covers the basics of setting up and using VisionPDF.

Prerequisites
~~~~~~~~~~~~

Before starting, ensure you have:

- Python 3.8 or higher installed
- A VLM backend (Ollama, llama.cpp, or custom API)
- Optional: OCR engines for fallback processing

Step 1: Installation
~~~~~~~~~~~~~~~~~~~~

Install VisionPDF and its dependencies:

.. code-block:: bash

   # Basic installation
   pip install visionpdf

   # With optional OCR support
   pip install visionpdf[ocr]

   # For development
   git clone https://github.com/your-repo/visionpdf.git
   cd visionpdf
   pip install -e .[dev]

Step 2: Backend Setup
~~~~~~~~~~~~~~~~~~~~

**Option A: Ollama (Recommended for beginners)**

.. code-block:: bash

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull a model
   ollama pull llama2

**Option B: llama.cpp**

.. code-block:: bash

   # Install llama.cpp (follow official guide)
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make

   # Download a model
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin

Step 3: Your First Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple Python script:

.. code-block:: python

   # first_conversion.py
   import asyncio
   from vision_pdf import VisionPDF

   async def main():
       # Initialize processor
       processor = VisionPDF()

       # Convert PDF to markdown
       try:
           markdown = await processor.convert_pdf("sample.pdf")
           print("Conversion successful!")
           print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
       except Exception as e:
           print(f"Error: {e}")

   if __name__ == "__main__":
       asyncio.run(main())

Run it:

.. code-block:: bash

   python first_conversion.py

Step 4: Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration script:

.. code-block:: python

   # advanced_setup.py
   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode

   def create_config():
       config = VisionPDFConfig()

       # Backend configuration
       config.default_backend = BackendType.OLLAMA
       config.backends[BackendType.OLLAMA.value].config = {
           "model": "llama2",
           "temperature": 0.1,
           "timeout": 60
       }

       # Processing configuration
       config.processing.mode = ProcessingMode.HYBRID
       config.processing.preserve_tables = True
       config.processing.preserve_math = True
       config.processing.preserve_code = True

       # OCR fallback
       config.processing.ocr_fallback_enabled = True
       config.processing.ocr_config = {
           "engine": "tesseract",
           "languages": ["eng"]
       }

       # Performance
       config.cache.enabled = True
       config.cache.max_size_mb = 512

       return config

   async def main():
       config = create_config()
       processor = VisionPDF(config=config)

       await processor.convert_pdf_to_file("sample.pdf", "output.md")
       print("Advanced conversion complete!")

   asyncio.run(main())

Tutorial 2: Processing Academic Papers
---------------------------------------

This tutorial focuses on converting academic papers with mathematical content.

Challenge
~~~~~~~~~

Academic papers contain:
- Complex mathematical formulas
- Tables and figures
- Citations and references
- Multi-column layouts
- Headers and footers

Solution
~~~~~~~~

.. code-block:: python

   # academic_paper_processor.py
   import asyncio
   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode

   async def process_academic_paper():
       # Configure for academic content
       config = VisionPDFConfig()

       # Use a model good at academic content
       config.default_backend = BackendType.OLLAMA
       config.backends[BackendType.OLLAMA.value].config = {
           "model": "codellama",  # Good for academic/technical content
           "temperature": 0.1,    # Lower temperature for consistency
           "top_p": 0.9
       }

       # Processing mode
       config.processing.mode = ProcessingMode.HYBRID
       config.processing.preserve_math = True
       config.processing.preserve_tables = True
       config.processing.preserve_code = True

       # OCR fallback for scanned pages
       config.processing.ocr_fallback_enabled = True
       config.processing.ocr_config = {
           "engine": "tesseract",
           "languages": ["eng"],
           "preprocessing": True,
           "deskew": True
       }

       # Custom prompt for academic papers
       config.processing.custom_prompt = (
           "Convert this academic paper page to markdown. "
           "Pay special attention to: "
           "1. Mathematical expressions - convert to LaTeX format "
           "2. Tables - preserve exact structure and data "
           "3. Citations - maintain citation format "
           "4. Section headers - preserve hierarchy "
           "5. Figures and tables - include captions "
           "6. Multi-column layouts - read in correct order"
       )

       processor = VisionPDF(config=config)

       # Process with progress monitoring
       def progress_callback(current, total, filename):
           percent = (current / total) * 100
           print(f"Processing page {current}/{total} ({percent:.1f}%)")

       try:
           markdown = await processor.convert_pdf(
               "research_paper.pdf",
               progress_callback=progress_callback
           )

           # Save result
           with open("paper_markdown.md", "w", encoding="utf-8") as f:
               f.write(markdown)

           print("Academic paper converted successfully!")
           return markdown

       except Exception as e:
           print(f"Error processing academic paper: {e}")
           raise

   # Run the processor
   if __name__ == "__main__":
       result = asyncio.run(process_academic_paper())

Best Practices for Academic Papers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use appropriate models**: Code-focused models like CodeLlama often handle technical content well
2. **Enable OCR fallback**: Academic papers often have scanned figures or pages
3. **Preserve mathematical expressions**: Ensure LaTeX conversion is enabled
4. **Handle citations**: Custom prompts help maintain citation formats
5. **Multi-column awareness**: Some models need explicit instructions for reading order

Tutorial 3: API Documentation Processing
-----------------------------------------

This tutorial covers converting API documentation and technical specifications.

Challenge
~~~~~~~~~

API documentation typically contains:
- Code examples and snippets
- Configuration tables
- API endpoints and parameters
- JSON/XML examples
- Error codes and messages

Solution
~~~~~~~~

.. code-block:: python

   # api_docs_processor.py
   import asyncio
   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType
   from vision_pdf.markdown.formatters.code import CodeLanguage

   async def process_api_documentation():
       # Configure for technical documentation
       config = VisionPDFConfig()

       # Use code-focused model
       config.default_backend = BackendType.OLLAMA
       config.backends[BackendType.OLLAMA.value].config = {
           "model": "codellama:7b-code",
           "temperature": 0.05,  # Very low for code consistency
           "num_ctx": 4096      # Larger context for complex docs
       }

       # Enable all content preservation
       config.processing.preserve_code = True
       config.processing.preserve_tables = True
       config.processing.preserve_images = True  # For diagrams
       config.processing.mode = ProcessingMode.HYBRID

       # Enhanced OCR for code snippets
       config.processing.ocr_fallback_enabled = True
       config.processing.ocr_config = {
           "engine": "tesseract",
           "languages": ["eng"],
           "confidence_threshold": 0.7,
           "preprocessing": True
       }

       # Custom prompt for API docs
       config.processing.custom_prompt = (
           "Convert this API documentation page to markdown. "
           "Focus on preserving: "
           "1. Code examples with exact syntax and indentation "
           "2. API endpoints with proper formatting "
           "3. Parameter tables with all details "
           "4. JSON/XML examples with correct structure "
           "5. HTTP status codes and error messages "
           "6. Code blocks with appropriate language highlighting"
       )

       processor = VisionPDF(config=config)

       try:
           # Process with performance monitoring
           from vision_pdf.utils.performance import PerformanceOptimizer

           optimizer = PerformanceOptimizer({
               'monitoring_enabled': True,
               'cache_enabled': True,
               'parallel_processing': True,
               'max_workers': 4
           })

           with optimizer.measure_operation("api_docs_conversion"):
               markdown = await processor.convert_pdf("api_documentation.pdf")

           # Post-process for additional formatting
           markdown = post_process_api_docs(markdown)

           # Save result
           with open("api_docs_markdown.md", "w", encoding="utf-8") as f:
               f.write(markdown)

           print("API documentation converted successfully!")

           # Print performance stats
           stats = optimizer.performance_monitor.get_summary()
           print(f"Performance: {stats['total_operations']} operations, "
                 f"{stats['overall_success_rate']:.1%} success rate")

           return markdown

       except Exception as e:
           print(f"Error processing API documentation: {e}")
           raise

   def post_process_api_docs(markdown: str) -> str:
       """Additional post-processing for API documentation."""
       # Add consistent formatting
       lines = markdown.split('\n')
       processed_lines = []

       for line in lines:
           # Enhance code block language detection
           if line.strip().startswith('```'):
               # Try to detect language from content
               if 'curl' in markdown.lower():
                   processed_lines.append('```bash')
               elif 'json' in markdown.lower():
                   processed_lines.append('```json')
               elif 'python' in markdown.lower():
                   processed_lines.append('```python')
               else:
                   processed_lines.append(line)
           else:
               processed_lines.append(line)

       return '\n'.join(processed_lines)

   # Run the processor
   if __name__ == "__main__":
       result = asyncio.run(process_api_documentation())

Batch Processing Multiple API Docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # batch_api_docs.py
   import asyncio
   from pathlib import Path

   async def batch_process_api_docs():
       config = VisionPDFConfig()
       config.default_backend = BackendType.OLLAMA
       config.backends[BackendType.OLLAMA.value].config = {
           "model": "codellama:7b-code"
       }
       config.processing.preserve_code = True
       config.processing.preserve_tables = True

       processor = VisionPDF(config=config)

       # Find all PDF files in docs directory
       docs_dir = Path("api_docs")
       pdf_files = list(docs_dir.glob("*.pdf"))

       print(f"Found {len(pdf_files)} API documentation files")

       def progress_callback(current, total, filename):
           percent = (current / total) * 100
           print(f"Converting {filename} ({percent:.1f}%)")

       # Process all files
       output_dir = Path("converted_docs")
       results = await processor.convert_batch(
           [str(f) for f in pdf_files],
           str(output_dir),
           progress_callback=progress_callback
       )

       print(f"Successfully converted {len(results)} files")
       return results

   if __name__ == "__main__":
       asyncio.run(batch_process_api_docs())

Tutorial 4: Financial Document Processing
-----------------------------------------

This tutorial demonstrates processing financial reports with complex tables and numbers.

Challenge
~~~~~~~~~

Financial documents contain:
- Complex tables with numerical data
- Financial statements
- Charts and graphs
- Regulatory information
- Multi-page reports

Solution
~~~~~~~~

.. code-block:: python

   # financial_docs_processor.py
   import asyncio
   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType
   from vision_pdf.utils.performance import PerformanceOptimizer
   import re

   async def process_financial_report():
       # Configure for financial content
       config = VisionPDFConfig()

       config.default_backend = BackendType.OLLAMA
       config.backends[BackendType.OLLAMA.value].config = {
           "model": "llama2",  # Good general model
           "temperature": 0.1,
           "top_p": 0.95
       }

       # Focus on table preservation
       config.processing.preserve_tables = True
       config.processing.preserve_math = True  # For financial formulas
       config.processing.mode = ProcessingMode.HYBRID

       # High-quality OCR for financial data
       config.processing.ocr_fallback_enabled = True
       config.processing.ocr_fallback_threshold = 0.4  # Lower threshold
       config.processing.ocr_config = {
           "engine": "tesseract",
           "languages": ["eng"],
           "confidence_threshold": 0.6,
           "preprocessing": True,
           "deskew": True,
           "enhancement": True
       }

       # Custom prompt for financial documents
       config.processing.custom_prompt = (
           "Convert this financial document to markdown. "
           "Pay special attention to: "
           "1. Numerical tables - preserve all numbers and formatting "
           "2. Financial statements - maintain structure "
           "3. Currency symbols and amounts "
           "4. Percentages and ratios "
           "5. Dates and periods "
           "6. Chart titles and legends"
       )

       processor = VisionPDF(config=config)

       try:
           # Process with monitoring
           optimizer = PerformanceOptimizer({
               'monitoring_enabled': True,
               'cache_enabled': True,
               'resource_monitoring_enabled': True
           })

           with optimizer.measure_operation("financial_processing"):
               with optimizer.resource_monitor.monitor_memory("financial_doc"):
                   markdown = await processor.convert_pdf("financial_report.pdf")

           # Post-process financial content
           markdown = enhance_financial_content(markdown)

           # Save with metadata
           output_file = "financial_report_processed.md"
           with open(output_file, "w", encoding="utf-8") as f:
               # Add processing metadata
               f.write("<!-- Financial Report Processing Summary -->\n")
               f.write(f"<!-- Processed: {asyncio.get_event_loop().time()} -->\n")
               f.write(f"<!-- Pages: {markdown.count('## Page')} -->\n")
               f.write(f"<!-- Tables: {markdown.count('|')} -->\n")
               f.write("\n---\n\n")
               f.write(markdown)

           print(f"Financial report processed and saved to {output_file}")

           # Generate performance report
           report = optimizer.generate_performance_report("financial_performance.json")
           print(f"Performance report saved")

           return markdown

       except Exception as e:
           print(f"Error processing financial report: {e}")
           raise

   def enhance_financial_content(markdown: str) -> str:
       """Enhance financial content formatting."""

       # Improve number formatting
       def enhance_numbers(text):
           # Add thousand separators for large numbers
           def add_separators(match):
               num = match.group()
               try:
                   value = int(num.replace(',', ''))
                   return f"{value:,}"
               except:
                   return num

           # Pattern for numbers (with optional decimals)
           number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
           return re.sub(number_pattern, add_separators, text)

       # Enhance currency formatting
       def enhance_currency(text):
           # Pattern for currency amounts
           currency_pattern = r'(\$|€|£|¥)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
           return re.sub(currency_pattern, r'\1\2', text)

       # Apply enhancements
       enhanced = enhance_numbers(markdown)
       enhanced = enhance_currency(enhanced)

       return enhanced

   # Run the processor
   if __name__ == "__main__":
       result = asyncio.run(process_financial_report())

Validation and Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # quality_checks.py
   import re
   from typing import Dict, List

   def validate_financial_conversion(markdown: str) -> Dict[str, any]:
       """Validate financial document conversion quality."""

       results = {
           "tables_detected": 0,
           "numbers_found": 0,
           "currency_amounts": 0,
           "percentages": 0,
           "dates": 0,
           "issues": []
       }

       # Count tables
       table_matches = re.findall(r'\|.*\|', markdown)
       results["tables_detected"] = len(table_matches) // 2  # Approximate

       # Count numbers
       number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
       number_matches = re.findall(number_pattern, markdown)
       results["numbers_found"] = len(number_matches)

       # Count currency amounts
       currency_pattern = r'(\$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d{2})?'
       currency_matches = re.findall(currency_pattern, markdown)
       results["currency_amounts"] = len(currency_matches)

       # Count percentages
       percent_pattern = r'\d+(?:\.\d+)?%'
       percent_matches = re.findall(percent_pattern, markdown)
       results["percentages"] = len(percent_matches)

       # Check for common issues
       if results["tables_detected"] == 0:
           results["issues"].append("No tables detected - possible OCR failure")

       if results["numbers_found"] < 10:
           results["issues"].append("Very few numbers found - possible processing issue")

       return results

   # Usage
   def validate_conversion(markdown_content):
       validation = validate_financial_conversion(markdown_content)

       print("Financial Document Validation:")
       print(f"Tables detected: {validation['tables_detected']}")
       print(f"Numbers found: {validation['numbers_found']}")
       print(f"Currency amounts: {validation['currency_amounts']}")
       print(f"Percentages: {validation['percentages']}")

       if validation['issues']:
           print("\nIssues found:")
           for issue in validation['issues']:
               print(f"  - {issue}")
       else:
           print("\n✓ No major issues detected")

Tutorial 5: Batch Processing Workflow
------------------------------------

This tutorial covers setting up automated batch processing for large document collections.

Scenario
~~~~~~~~~

You have a directory of PDF documents that need to be processed regularly:
- Different document types (academic papers, reports, manuals)
- Various quality levels
- Need for consistent output
- Performance monitoring

Solution
~~~~~~~~

.. code-block:: python

   # batch_workflow.py
   import asyncio
   import json
   from pathlib import Path
   from datetime import datetime
   from typing import Dict, List, Any
   from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode
   from vision_pdf.utils.performance import PerformanceOptimizer

   class BatchProcessor:
       """Advanced batch processing system."""

       def __init__(self, config_path: str = None):
           """Initialize batch processor."""
           self.config = self.load_config(config_path)
           self.setup_processor()
           self.setup_monitoring()
           self.results = []

       def load_config(self, config_path: str) -> Dict[str, Any]:
           """Load configuration from file."""
           if config_path and Path(config_path).exists():
               with open(config_path, 'r') as f:
                   return json.load(f)

           # Default configuration
           return {
               "backend": {
                   "type": "ollama",
                   "model": "llama2",
                   "temperature": 0.1
               },
               "processing": {
                   "mode": "hybrid",
                   "preserve_tables": True,
                   "preserve_math": True,
                   "preserve_code": True,
                   "ocr_fallback_enabled": True
               },
               "performance": {
                   "parallel_processing": True,
                   "max_workers": 4,
                   "batch_size": 5,
                   "cache_enabled": True,
                   "cache_size_mb": 1024
               },
               "output": {
                   "include_metadata": True,
                   "create_index": True,
                   "validate_output": True
               }
           }

       def setup_processor(self):
           """Setup VisionPDF processor."""
           config = VisionPDFConfig()

           # Backend configuration
           backend_type = BackendType(self.config["backend"]["type"])
           config.default_backend = backend_type
           config.backends[backend_type.value].config = self.config["backend"]

           # Processing configuration
           proc_config = self.config["processing"]
           config.processing.mode = ProcessingMode(proc_config["mode"])
           config.processing.preserve_tables = proc_config["preserve_tables"]
           config.processing.preserve_math = proc_config["preserve_math"]
           config.processing.preserve_code = proc_config["preserve_code"]
           config.processing.ocr_fallback_enabled = proc_config["ocr_fallback_enabled"]

           # Performance configuration
           perf_config = self.config["performance"]
           config.processing.parallel_processing = perf_config["parallel_processing"]
           config.processing.max_workers = perf_config["max_workers"]
           config.processing.batch_size = perf_config["batch_size"]
           config.cache.enabled = perf_config["cache_enabled"]
           config.cache.max_size_mb = perf_config["cache_size_mb"]

           self.processor = VisionPDF(config=config)

       def setup_monitoring(self):
           """Setup performance monitoring."""
           self.optimizer = PerformanceOptimizer({
               'monitoring_enabled': True,
               'cache_enabled': self.config["performance"]["cache_enabled"],
               'parallel_processing': self.config["performance"]["parallel_processing"],
               'max_workers': self.config["processing"]["max_workers"],
               'resource_monitoring_enabled': True
           })

       async def process_directory(self, input_dir: str, output_dir: str):
           """Process all PDFs in a directory."""
           input_path = Path(input_dir)
           output_path = Path(output_dir)
           output_path.mkdir(parents=True, exist_ok=True)

           # Find all PDF files
           pdf_files = list(input_path.glob("**/*.pdf"))
           print(f"Found {len(pdf_files)} PDF files to process")

           if not pdf_files:
               print("No PDF files found")
               return

           # Process files
           start_time = datetime.now()

           def progress_callback(current, total, filename):
               percent = (current / total) * 100
               elapsed = (datetime.now() - start_time).total_seconds()
               rate = current / elapsed if elapsed > 0 else 0
               eta = (total - current) / rate if rate > 0 else 0

               print(f"Progress: {current}/{total} ({percent:.1f}%) "
                     f"- Rate: {rate:.1f} files/sec - ETA: {eta:.0f}s - {filename}")

           try:
               results = await self.processor.convert_batch(
                   [str(f) for f in pdf_files],
                   str(output_path),
                   progress_callback=progress_callback
               )

               # Post-process results
               await self.post_process_results(pdf_files, results, output_path)

               print(f"Successfully processed {len(results)} files")
               return results

           except Exception as e:
               print(f"Error during batch processing: {e}")
               raise
           finally:
               await self.processor.close()

       async def post_process_results(self, input_files: List[Path],
                                     output_files: List[str],
                                     output_dir: Path):
           """Post-process batch results."""

           # Create processing summary
           summary = {
               "timestamp": datetime.now().isoformat(),
               "input_files": len(input_files),
               "output_files": len(output_files),
               "processing_time": datetime.now().isoformat(),
               "configuration": self.config,
               "results": []
           }

           # Process each result
           for input_file, output_file in zip(input_files, output_files):
               if Path(output_file).exists():
                   result_info = {
                       "input_file": str(input_file),
                       "output_file": output_file,
                       "status": "success",
                       "file_size": Path(output_file).stat().st_size
                   }

                   # Validate output if enabled
                   if self.config["output"]["validate_output"]:
                       validation = self.validate_output(output_file)
                       result_info["validation"] = validation

                   summary["results"].append(result_info)
               else:
                   summary["results"].append({
                       "input_file": str(input_file),
                       "output_file": output_file,
                       "status": "failed"
                   })

           # Save summary
           summary_file = output_dir / "processing_summary.json"
           with open(summary_file, 'w') as f:
               json.dump(summary, f, indent=2)

           # Create index if enabled
           if self.config["output"]["create_index"]:
               await self.create_index(summary["results"], output_dir)

           # Generate performance report
           perf_report = self.optimizer.generate_performance_report(
               output_dir / "performance_report.json"
           )

           print(f"Processing summary saved to {summary_file}")
           print(f"Performance report saved to {output_dir / 'performance_report.json'}")

       def validate_output(self, output_file: str) -> Dict[str, Any]:
           """Validate output markdown file."""
           path = Path(output_file)

           if not path.exists():
               return {"valid": False, "error": "File does not exist"}

           try:
               with open(path, 'r', encoding='utf-8') as f:
                   content = f.read()

               validation = {
                   "valid": True,
                   "size_bytes": len(content.encode('utf-8')),
                   "line_count": len(content.split('\n')),
                   "has_headings": content.count('#') > 0,
                   "has_tables": content.count('|') > 4,
                   "has_code": '```' in content,
                   "has_math': '$' in content or '\\[' in content
               }

               return validation

           except Exception as e:
               return {"valid": False, "error": str(e)}

       async def create_index(self, results: List[Dict], output_dir: Path):
           """Create an index file for all processed documents."""

           index_content = ["# Document Index\n"]
           index_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
           index_content.append(f"Total Documents: {len(results)}\n")

           # Group by status
           successful = [r for r in results if r.get("status") == "success"]
           failed = [r for r in results if r.get("status") == "failed"]

           index_content.append(f"\n## Successfully Processed ({len(successful)})\n")

           for result in successful:
               input_name = Path(result["input_file"]).stem
               index_content.append(f"- [{input_name}]({Path(result['output_file']).name})")

               if "validation" in result:
                   validation = result["validation"]
                       if validation.get("has_tables"):
                           index_content.append("  - Contains tables")
                       if validation.get("has_code"):
                           index_content.append("  - Contains code blocks")
                       if validation.get("has_math"):
                           index_content.append("  - Contains math expressions")

           if failed:
               index_content.append(f"\n## Failed to Process ({len(failed)})\n")
               for result in failed:
                   input_name = Path(result["input_file"]).stem
                   index_content.append(f"- {input_name}")

           # Write index file
           index_file = output_dir / "index.md"
           with open(index_file, 'w', encoding='utf-8') as f:
               f.write('\n'.join(index_content))

           print(f"Index created at {index_file}")

   # Usage example
   async def main():
       processor = BatchProcessor("batch_config.json")

       await processor.process_directory(
           input_dir="input_documents",
           output_dir="processed_documents"
       )

   if __name__ == "__main__":
       asyncio.run(main())

Configuration File
~~~~~~~~~~~~~~~~~~

Create a `batch_config.json` file:

.. code-block:: json

   {
     "backend": {
       "type": "ollama",
       "model": "llama2",
       "temperature": 0.1,
       "timeout": 120
     },
     "processing": {
       "mode": "hybrid",
       "preserve_tables": true,
       "preserve_math": true,
       "preserve_code": true,
       "ocr_fallback_enabled": true,
       "ocr_config": {
         "engine": "tesseract",
         "languages": ["eng"],
         "confidence_threshold": 0.6
       }
     },
     "performance": {
       "parallel_processing": true,
       "max_workers": 6,
       "batch_size": 3,
       "cache_enabled": true,
       "cache_size_mb": 2048
     },
     "output": {
       "include_metadata": true,
       "create_index": true,
       "validate_output": true
     }
   }

This comprehensive batch processing system provides:

- **Configurable processing**: JSON-based configuration for easy customization
- **Progress monitoring**: Real-time progress with ETA calculation
- **Performance tracking**: Detailed performance metrics and reports
- **Quality validation**: Automatic validation of output quality
- **Error handling**: Robust error handling and reporting
- **Index generation**: Automatic index creation for processed documents
- **Scalability**: Optimized for large document collections

The batch processor can handle diverse document types while maintaining consistent quality and providing comprehensive reporting for monitoring and debugging.
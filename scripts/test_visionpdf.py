#!/usr/bin/env python3
"""
Test script for VisionPDF with Ollama qwen3-vl:2b model.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from vision_pdf import VisionPDF, VisionPDFConfig, BackendType, ProcessingMode
    print("✓ Successfully imported VisionPDF")
except ImportError as e:
    print(f"✗ Failed to import VisionPDF: {e}")
    print("Make sure you're in the project directory with all dependencies installed")
    sys.exit(1)


async def test_with_qwen_vl():
    """Test VisionPDF with qwen3-vl:2b model."""

    print("\n" + "="*60)
    print("🚀 Testing VisionPDF with qwen3-vl:2b model")
    print("="*60)

    # Configure VisionPDF for qwen3-vl:2b
    config = VisionPDFConfig()
    config.default_backend = BackendType.OLLAMA

    # Backend configuration for qwen3-vl:2b
    config.backends[BackendType.OLLAMA.value].config = {
        "model": "qwen3-vl:2b",
        "host": "http://localhost:11434",
        "temperature": 0.1,
        "timeout": 120
    }

    # Processing configuration
    config.processing.mode = ProcessingMode.HYBRID
    config.processing.preserve_tables = True
    config.processing.preserve_math = True
    config.processing.preserve_code = True
    config.processing.ocr_fallback_enabled = True

    # OCR fallback configuration
    config.processing.ocr_config = {
        "engine": "tesseract",
        "languages": ["eng"],
        "confidence_threshold": 0.6,
        "preprocessing": True
    }

    # Performance optimization
    config.cache.enabled = True
    config.cache.max_size_mb = 512
    config.processing.max_workers = 2  # Conservative for testing
    config.processing.batch_size = 3

    print(f"📋 Configuration:")
    print(f"   Model: qwen3-vl:2b")
    print(f"   Mode: {config.processing.mode.value}")
    print(f"   Table preservation: {config.processing.preserve_tables}")
    print(f"   Math preservation: {config.processing.preserve_math}")
    print(f"   Code preservation: {config.processing.preserve_code}")
    print(f"   OCR fallback: {config.processing.ocr_fallback_enabled}")

    # Initialize processor
    processor = VisionPDF(config=config, backend_type=BackendType.OLLAMA)

    try:
        # Test backend connection first
        print(f"\n🔍 Testing backend connection...")
        connection_ok = await processor.test_backend_connection()

        if not connection_ok:
            print(f"✗ Backend connection failed!")
            print(f"   Make sure Ollama is running: 'ollama serve'")
            print(f"   And qwen3-vl:2b model is available: 'ollama list'")
            return

        print(f"✓ Backend connection successful!")

        # Check available models
        try:
            models = await processor.get_available_models()
            print(f"📦 Available models: {models}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")

        # PDF file path
        pdf_path = "Pulse_of_the_Profession_2025 1.pdf"
        output_path = "Pulse_of_the_Profession_2025_output.md"

        print(f"\n📄 Processing PDF: {pdf_path}")
        print(f"💾 Output will be saved to: {output_path}")

        # Progress callback
        def progress_callback(current, total):
            percent = (current / total) * 100 if total > 0 else 0
            print(f"📊 Progress: {current}/{total} pages ({percent:.1f}%)")

        # Process the PDF
        print(f"\n⚡ Starting PDF conversion...")
        start_time = asyncio.get_event_loop().time()

        markdown = await processor.convert_pdf(
            pdf_path,
            progress_callback=progress_callback
        )

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"\n✅ Conversion completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📄 Output saved to: {output_path}")
        print(f"📊 Markdown size: {len(markdown)} characters")
        print(f"📊 Line count: {len(markdown.splitlines())} lines")

        # Show preview
        print(f"\n📋 Preview of converted content:")
        print("-" * 50)
        preview_lines = markdown.splitlines()[:20]
        for i, line in enumerate(preview_lines, 1):
            print(f"{i:2d}: {line}")

        if len(markdown.splitlines()) > 20:
            print(f"... (and {len(markdown.splitlines()) - 20} more lines)")

        print("-" * 50)

        # Analyze content
        print(f"\n🔍 Content Analysis:")
        has_tables = '|' in markdown and markdown.count('|') > 10
        has_code = '```' in markdown
        has_math = ('$' in markdown or r'\[' in markdown) and markdown.count('$') > 5
        has_headings = '#' in markdown

        print(f"   Tables detected: {'✓' if has_tables else '✗'}")
        print(f"   Code blocks detected: {'✓' if has_code else '✗'}")
        print(f"   Math expressions detected: {'✓' if has_math else '✗'}")
        print(f"   Headings detected: {'✓' if has_headings else '✗'}")

        return markdown

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print(f"\n🔧 Troubleshooting tips:")
        print(f"   1. Check if Ollama is running: 'ollama serve'")
        print(f"   2. Verify model exists: 'ollama list | grep qwen3-vl'")
        print(f"   3. Pull model if needed: 'ollama pull qwen3-vl:2b'")
        print(f"   4. Check PDF file exists and is readable")
        print(f"   5. Ensure sufficient disk space for output")
        raise

    finally:
        # Clean up resources
        try:
            await processor.close()
            print(f"\n🧹 Resources cleaned up")
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")


async def check_prerequisites():
    """Check system prerequisites before running."""
    print("🔍 Checking prerequisites...")

    # Check if PDF file exists
    pdf_path = Path("Pulse_of_the_Profession_2025 1.pdf")
    if not pdf_path.exists():
        print(f"✗ PDF file not found: {pdf_path}")
        print(f"   Make sure the PDF file is in the current directory")
        return False

    print(f"✓ PDF file found: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try to import required modules
    try:
        import ollama
        print(f"✓ Ollama Python library available")
    except ImportError:
        print(f"⚠️  Ollama Python library not found")
        print(f"   Install with: pip install ollama")
        return False

    # Check if Ollama is running
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ Ollama service is running")

            # Check if qwen3-vl:2b is available
            if 'qwen3-vl:2b' in result.stdout:
                print(f"✓ qwen3-vl:2b model is available")
            else:
                print(f"⚠️  qwen3-vl:2b model not found in Ollama")
                print(f"   Pull with: ollama pull qwen3-vl:2b")
                return False
        else:
            print(f"✗ Ollama service not responding")
            print(f"   Start with: ollama serve")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"✗ Could not check Ollama status: {e}")
        print(f"   Make sure Ollama is installed and running")
        return False

    print(f"✓ All prerequisites checked")
    return True


def main():
    """Main entry point."""
    print("🎯 VisionPDF Test with qwen3-vl:2b Model")
    print("=" * 50)

    # Check prerequisites
    if not asyncio.run(check_prerequisites()):
        print(f"\n❌ Prerequisites not met. Please fix the issues above.")
        return 1

    try:
        # Run the test
        asyncio.run(test_with_qwen_vl())
        print(f"\n🎉 Test completed successfully!")
        return 0

    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
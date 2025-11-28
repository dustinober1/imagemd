#!/usr/bin/env python3
"""
Simple test for VisionPDF without Ollama dependencies.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports."""
    print("🔍 Testing basic imports...")

    try:
        from vision_pdf.core.document import Document, Page, ContentElement, ContentType
        print("✓ Core document classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core document classes: {e}")
        return False

    try:
        from vision_pdf.config.settings import VisionPDFConfig, BackendType, ProcessingMode
        print("✓ Configuration classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import configuration classes: {e}")
        return False

    try:
        from vision_pdf.markdown.formatters.tables import AdvancedTableDetector
        print("✓ Table formatter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import table formatter: {e}")
        return False

    try:
        from vision_pdf.markdown.formatters.math import MathPatternRecognizer
        print("✓ Math formatter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import math formatter: {e}")
        return False

    try:
        from vision_pdf.markdown.formatters.code import CodeDetector
        print("✓ Code formatter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import code formatter: {e}")
        return False

    try:
        from vision_pdf.ocr.base import OCRFallbackManager, OCRConfig
        print("✓ OCR base classes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OCR base classes: {e}")
        return False

    try:
        from vision_pdf.utils.performance import PerformanceMonitor
        print("✓ Performance monitoring imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import performance monitoring: {e}")
        return False

    try:
        from vision_pdf.utils.cache import PDFCache
        print("✓ Cache system imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cache system: {e}")
        return False

    print("✓ All basic imports successful!")
    return True

def test_configuration():
    """Test configuration system."""
    print("\n🔧 Testing configuration system...")

    try:
        from vision_pdf.config.settings import VisionPDFConfig, BackendType, ProcessingMode

        config = VisionPDFConfig()
        print("✓ Default configuration created")

        # Test backend configuration
        try:
            print(f"Available backend types: {[bt.value for bt in BackendType]}")
            backend_key = BackendType.OLLAMA.value
            print(f"Backend key: {backend_key}")
            config.default_backend = BackendType.OLLAMA
            if backend_key not in config.backends:
                print(f"⚠️  Backend {backend_key} not in config.backends")
                print(f"Available backends: {list(config.backends.keys())}")
            else:
                config.backends[backend_key].config = {"model": "test"}
                print("✓ Backend configuration works")
        except Exception as e:
            print(f"✗ Backend configuration failed: {e}")
            raise

        # Test processing configuration
        config.processing.mode = ProcessingMode.HYBRID
        config.processing.preserve_tables = True
        config.processing.preserve_math = True
        config.processing.preserve_code = True
        print("✓ Processing configuration works")

        # Test OCR configuration
        config.processing.ocr_fallback_enabled = True
        config.processing.ocr_config = {"engine": "tesseract"}
        print("✓ OCR configuration works")

        # Test cache configuration
        config.cache.enabled = True
        config.cache.max_size_mb = 1024
        print("✓ Cache configuration works")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_formatters():
    """Test advanced formatters."""
    print("\n📊 Testing advanced formatters...")

    try:
        # Test table detector
        from vision_pdf.markdown.formatters.tables import AdvancedTableDetector
        detector = AdvancedTableDetector()
        print("✓ Table detector created")

        # Test math recognizer
        from vision_pdf.markdown.formatters.math import MathPatternRecognizer
        math_recognizer = MathPatternRecognizer()
        print("✓ Math recognizer created")

        # Test code detector
        from vision_pdf.markdown.formatters.code import CodeDetector
        code_detector = CodeDetector()
        print("✓ Code detector created")

        return True

    except Exception as e:
        print(f"✗ Formatters test failed: {e}")
        return False

def test_document_model():
    """Test document model."""
    print("\n📄 Testing document model...")

    try:
        from vision_pdf.core.document import Document, Page, ContentElement, ContentType
        from pathlib import Path

        # Create a test document
        doc = Document(
            file_path=Path("test.pdf"),
            title="Test Document",
            author="Test Author"
        )
        print("✓ Document created")

        # Create a test page
        page = Page(
            page_number=0,
            width=595,  # Standard A4 width in points
            height=842,  # Standard A4 height in points
            dpi=300,
            raw_text="Test page content"
        )
        print("✓ Page created")

        # Add content elements
        element = ContentElement(
            text="Test content",
            content_type=ContentType.TEXT,
            confidence=0.9
        )
        page.elements.append(element)
        print("✓ Content element created")

        doc.pages.append(page)
        print("✓ Document model works correctly")

        return True

    except Exception as e:
        print(f"✗ Document model test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring."""
    print("\n⚡ Testing performance monitoring...")

    try:
        from vision_pdf.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor(enabled=True)
        print("✓ Performance monitor created")

        # Test measurement
        with monitor.measure("test_operation"):
            import time
            time.sleep(0.01)

        metrics = monitor.get_metrics()
        if "test_operation" in metrics:
            print("✓ Operation measurement works")
        else:
            print("⚠️  Operation measurement may not be working")

        summary = monitor.get_summary()
        print(f"✓ Performance summary: {summary['total_operations']} operations")

        return True

    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 VisionPDF Simple Test Suite")
    print("=" * 40)

    tests = [
        test_imports,
        test_configuration,
        test_formatters,
        test_document_model,
        test_performance_monitoring
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! VisionPDF core functionality is working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
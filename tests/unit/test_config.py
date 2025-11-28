"""
Unit tests for VisionPDF configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from vision_pdf.config.settings import (
    VisionPDFConfig,
    BackendConfig,
    ProcessingConfig,
    OCRConfig,
    CacheConfig,
    LoggingConfig,
    BackendType,
    ProcessingMode,
    CacheType,
    LogLevel
)


class TestVisionPDFConfig:
    """Test cases for VisionPDFConfig."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = VisionPDFConfig()

        assert config.processing.mode == ProcessingMode.HYBRID
        assert config.processing.parallel_processing is True
        assert config.processing.dpi == 300
        assert config.default_backend == BackendType.OLLAMA
        assert config.ocr.enabled is True
        assert config.cache.type == CacheType.MEMORY
        assert config.logging.level == LogLevel.INFO

    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_data = {
            "processing": {
                "mode": "vision_only",
                "dpi": 200,
                "parallel_processing": False
            },
            "default_backend": "llama_cpp",
            "ocr": {
                "enabled": False,
                "engine": "tesseract"
            }
        }

        config = VisionPDFConfig.load_from_dict(config_data)

        assert config.processing.mode == ProcessingMode.VISION_ONLY
        assert config.processing.dpi == 200
        assert config.processing.parallel_processing is False
        assert config.default_backend == BackendType.LLAMA_CPP
        assert config.ocr.enabled is False
        assert config.ocr.engine == "tesseract"

    def test_config_from_file(self, temp_dir):
        """Test creating configuration from YAML file."""
        config_data = {
            "processing": {
                "mode": "hybrid",
                "dpi": 150
            },
            "cache": {
                "type": "file",
                "directory": "/tmp/cache"
            }
        }

        config_path = temp_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        config = VisionPDFConfig.load_from_file(config_path)

        assert config.processing.mode == ProcessingMode.HYBRID
        assert config.processing.dpi == 150
        assert config.cache.type == CacheType.FILE
        assert config.cache.directory == "/tmp/cache"

    def test_config_from_env(self, monkeypatch):
        """Test creating configuration from environment variables."""
        monkeypatch.setenv("VISIONPDF_PROCESSING_MODE", "text_only")
        monkeypatch.setenv("VISIONPDF_PROCESSING_DPI", "200")
        monkeypatch.setenv("VISIONPDF_DEFAULT_BACKEND", "custom_api")
        monkeypatch.setenv("VISIONPDF_LOG_LEVEL", "DEBUG")

        config = VisionPDFConfig.load_from_env()

        assert config.processing.mode == ProcessingMode.TEXT_ONLY
        assert config.processing.dpi == 200
        assert config.default_backend == BackendType.CUSTOM_API
        assert config.logging.level == LogLevel.DEBUG

    def test_save_config_to_file(self, temp_dir, sample_config):
        """Test saving configuration to file."""
        config_path = temp_dir / "saved_config.yaml"

        sample_config.save_to_file(config_path)

        assert config_path.exists()

        # Load and verify
        loaded_config = VisionPDFConfig.load_from_file(config_path)
        assert loaded_config.processing.mode == sample_config.processing.mode
        assert loaded_config.processing.dpi == sample_config.processing.dpi
        assert loaded_config.default_backend == sample_config.default_backend

    def test_get_backend_config(self, sample_config):
        """Test getting backend configuration."""
        ollama_config = sample_config.get_backend_config(BackendType.OLLAMA)
        assert ollama_config is not None
        assert ollama_config.backend_type == BackendType.OLLAMA
        assert ollama_config.enabled is True

        # Test with string
        ollama_config_str = sample_config.get_backend_config("ollama")
        assert ollama_config_str is not None
        assert ollama_config_str.backend_type == BackendType.OLLAMA

        # Test non-existent backend
        nonexistent = sample_config.get_backend_config("nonexistent")
        assert nonexistent is None

    def test_is_backend_enabled(self, sample_config):
        """Test checking if backend is enabled."""
        assert sample_config.is_backend_enabled(BackendType.OLLAMA) is True
        assert sample_config.is_backend_enabled("llama_cpp") is True

        # Disable a backend
        sample_config.backends["ollama"].enabled = False
        assert sample_config.is_backend_enabled(BackendType.OLLAMA) is False

    def test_validation_errors(self):
        """Test configuration validation errors."""
        # Invalid DPI
        with pytest.raises(ValueError):
            VisionPDFConfig(processing={"dpi": 50})  # Below minimum

        # Invalid confidence threshold
        with pytest.raises(ValueError):
            VisionPDFConfig(ocr={"confidence_threshold": 1.5})  # Above maximum

        # Invalid cache directory for file-based caching
        with pytest.raises(ValueError):
            VisionPDFConfig(cache={"type": "file"})  # Missing directory


class TestProcessingConfig:
    """Test cases for ProcessingConfig."""

    def test_processing_config_defaults(self):
        """Test default processing configuration."""
        config = ProcessingConfig()

        assert config.mode == ProcessingMode.HYBRID
        assert config.parallel_processing is True
        assert config.max_workers == 4
        assert config.dpi == 300
        assert config.preserve_tables is True
        assert config.preserve_math is True
        assert config.preserve_code is True

    def test_processing_config_validation(self):
        """Test processing configuration validation."""
        # Valid configuration
        config = ProcessingConfig(
            mode=ProcessingMode.VISION_ONLY,
            max_workers=8,
            dpi=200,
            preserve_images=True
        )
        assert config.max_workers == 8
        assert config.dpi == 200
        assert config.preserve_images is True

        # Invalid max_workers
        with pytest.raises(ValueError):
            ProcessingConfig(max_workers=0)  # Below minimum

        with pytest.raises(ValueError):
            ProcessingConfig(max_workers=20)  # Above maximum

        # Invalid DPI
        with pytest.raises(ValueError):
            ProcessingConfig(dpi=50)  # Below minimum


class TestOCRConfig:
    """Test cases for OCRConfig."""

    def test_ocr_config_defaults(self):
        """Test default OCR configuration."""
        config = OCRConfig()

        assert config.engine == "easyocr"
        assert config.languages == ["en"]
        assert config.confidence_threshold == 0.5
        assert config.enabled is True

    def test_ocr_config_validation(self):
        """Test OCR configuration validation."""
        # Valid configuration
        config = OCRConfig(
            engine="tesseract",
            languages=["en", "fr", "de"],
            confidence_threshold=0.8
        )
        assert config.engine == "tesseract"
        assert config.languages == ["en", "fr", "de"]
        assert config.confidence_threshold == 0.8

        # Invalid confidence threshold
        with pytest.raises(ValueError):
            OCRConfig(confidence_threshold=1.5)  # Above maximum

        with pytest.raises(ValueError):
            OCRConfig(confidence_threshold=-0.1)  # Below minimum

        # Empty languages
        with pytest.raises(ValueError):
            OCRConfig(languages=[])  # Must have at least one language


class TestCacheConfig:
    """Test cases for CacheConfig."""

    def test_cache_config_defaults(self):
        """Test default cache configuration."""
        config = CacheConfig()

        assert config.type == CacheType.MEMORY
        assert config.directory is None
        assert config.max_size == 1000
        assert config.ttl == 3600

    def test_cache_config_validation(self):
        """Test cache configuration validation."""
        # File-based cache with directory
        config = CacheConfig(
            type=CacheType.FILE,
            directory="/tmp/test_cache",
            max_size=500
        )
        assert config.type == CacheType.FILE
        assert config.directory == "/tmp/test_cache"
        assert config.max_size == 500

        # File-based cache without directory should fail validation
        with pytest.raises(ValueError):
            CacheConfig(type=CacheType.FILE)  # Missing directory

        # Invalid sizes
        with pytest.raises(ValueError):
            CacheConfig(max_size=-1)  # Negative size

        with pytest.raises(ValueError):
            CacheConfig(ttl=-1)  # Negative TTL


class TestLoggingConfig:
    """Test cases for LoggingConfig."""

    def test_logging_config_defaults(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == LogLevel.INFO
        assert "%(asctime)s" in config.format
        assert config.file is None
        assert config.max_size == 10485760  # 10MB
        assert config.backup_count == 5

    def test_logging_config_validation(self):
        """Test logging configuration validation."""
        # Valid configuration
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file="/tmp/test.log",
            max_size=5242880,  # 5MB
            backup_count=3
        )
        assert config.level == LogLevel.DEBUG
        assert config.file == "/tmp/test.log"
        assert config.max_size == 5242880
        assert config.backup_count == 3

        # Invalid sizes
        with pytest.raises(ValueError):
            LoggingConfig(max_size=-1)  # Negative size

        with pytest.raises(ValueError):
            LoggingConfig(backup_count=-1)  # Negative backup count

    def test_log_file_directory_validation(self, temp_dir):
        """Test log file directory validation."""
        # Valid log file with existing directory
        log_file = temp_dir / "test.log"
        config = LoggingConfig(file=str(log_file))
        assert config.file == str(log_file)

        # Invalid log file with non-existing directory
        invalid_log_file = "/non/existing/path/test.log"
        with pytest.raises(ValueError):
            LoggingConfig(file=invalid_log_file)
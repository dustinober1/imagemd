"""
Configuration management system for VisionPDF.

This module provides comprehensive configuration management with support for
hierarchical settings, environment variable overrides, and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum

from ..backends.base import BackendType, ProcessingMode


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheType(str, Enum):
    """Supported cache types."""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"
    DISABLED = "disabled"


@dataclass
class BackendConfig:
    """Configuration for a specific backend."""
    backend_type: BackendType
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


class OCRConfig(BaseModel):
    """Configuration for OCR processing."""
    engine: str = Field(default="easyocr", description="OCR engine to use")
    languages: List[str] = Field(default=["en"], description="List of language codes")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = Field(default=True, description="Enable OCR processing")

    @validator('languages')
    def validate_languages(cls, v):
        """Validate language codes."""
        if not v:
            raise ValueError("At least one language must be specified")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    mode: ProcessingMode = Field(default=ProcessingMode.HYBRID)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=16)
    batch_size: int = Field(default=5, ge=1, le=50)
    dpi: int = Field(default=300, ge=72, le=600)
    preserve_tables: bool = Field(default=True)
    preserve_math: bool = Field(default=True)
    preserve_code: bool = Field(default=True)
    preserve_images: bool = Field(default=False)

    # OCR fallback configuration
    ocr_fallback_enabled: bool = Field(default=True, description="Enable OCR fallback for failed VLM processing")
    ocr_fallback_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold for OCR fallback")
    ocr_config: Optional[Dict[str, Any]] = Field(default=None, description="OCR engine configuration")

    @validator('dpi')
    def validate_dpi(cls, v):
        """Validate DPI range."""
        if v < 72 or v > 600:
            raise ValueError("DPI must be between 72 and 600")
        return v


class CacheConfig(BaseModel):
    """Configuration for result caching."""
    type: CacheType = Field(default=CacheType.MEMORY)
    directory: Optional[str] = Field(default=None, description="Cache directory for file-based caching")
    max_size: int = Field(default=1000, ge=0, description="Maximum number of cached items")
    ttl: int = Field(default=3600, ge=0, description="Cache time-to-live in seconds")

    @model_validator(mode='after')
    def validate_cache_directory(self):
        """Validate cache directory when file-based caching is used."""
        if self.type == CacheType.FILE and not self.directory:
            raise ValueError("Cache directory must be specified for file-based caching")
        return self


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: LogLevel = Field(default=LogLevel.INFO)
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file: Optional[str] = Field(default=None, description="Log file path")
    max_size: int = Field(default=10485760, ge=0, description="Maximum log file size in bytes")
    backup_count: int = Field(default=5, ge=0, description="Number of backup log files")

    @validator('file')
    def validate_log_file(cls, v, values):
        """Validate log file path."""
        if v:
            path = Path(v)
            if not path.parent.exists():
                raise ValueError(f"Log file directory does not exist: {path.parent}")
        return v


class VisionPDFConfig(BaseModel):
    """Main configuration class for VisionPDF."""

    # Core processing settings
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Backend configurations
    backends: Dict[str, BackendConfig] = Field(default_factory=dict)
    default_backend: BackendType = Field(default=BackendType.OLLAMA)

    # OCR settings
    ocr: OCRConfig = Field(default_factory=OCRConfig)

    # Cache settings
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Logging settings
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # General settings
    temp_directory: str = Field(default="/tmp/vision_pdf", description="Temporary files directory")
    max_file_size: int = Field(default=52428800, ge=0, description="Maximum file size in bytes (50MB default)")

    @validator('temp_directory')
    def validate_temp_directory(cls, v):
        """Validate temporary directory exists or can be created."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f"Cannot create temp directory: {v}")
        return str(path.absolute())

    @validator('backends', always=True, pre=True)
    def validate_backends(cls, v, values):
        """Validate backend configurations."""
        if not v:
            # Add default backend configurations
            v = {
                'ollama': BackendConfig(
                    backend_type=BackendType.OLLAMA,
                    config={'base_url': 'http://localhost:11434'}
                ),
                'llama_cpp': BackendConfig(
                    backend_type=BackendType.LLAMA_CPP,
                    config={'base_url': 'http://localhost:8080'}
                ),
                'custom_api': BackendConfig(
                    backend_type=BackendType.CUSTOM_API,
                    config={'base_url': 'https://api.example.com'}
                )
            }
        return v

    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "VisionPDFConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            VisionPDFConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Convert backend configurations
        if 'backends' in config_data:
            backends = {}
            for name, backend_data in config_data['backends'].items():
                backend_type = BackendType(backend_data['backend_type'])
                backends[name] = BackendConfig(
                    backend_type=backend_type,
                    enabled=backend_data.get('enabled', True),
                    config=backend_data.get('config', {}),
                    timeout=backend_data.get('timeout', 60.0),
                    max_retries=backend_data.get('max_retries', 3),
                    retry_delay=backend_data.get('retry_delay', 1.0)
                )
            config_data['backends'] = backends

        return cls(**config_data)

    @classmethod
    def load_from_dict(cls, config_data: Dict[str, Any]) -> "VisionPDFConfig":
        """
        Load configuration from a dictionary.

        Args:
            config_data: Configuration dictionary

        Returns:
            VisionPDFConfig instance
        """
        return cls(**config_data)

    @classmethod
    def load_from_env(cls) -> "VisionPDFConfig":
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with VISIONPDF_

        Returns:
            VisionPDFConfig instance
        """
        config_data = {}

        # Map environment variables to config keys
        env_mappings = {
            'VISIONPDF_PROCESSING_MODE': ('processing', 'mode'),
            'VISIONPDF_PROCESSING_PARALLEL': ('processing', 'parallel_processing'),
            'VISIONPDF_PROCESSING_DPI': ('processing', 'dpi'),
            'VISIONPDF_DEFAULT_BACKEND': ('default_backend', None),
            'VISIONPDF_OCR_ENGINE': ('ocr', 'engine'),
            'VISIONPDF_OCR_ENABLED': ('ocr', 'enabled'),
            'VISIONPDF_CACHE_TYPE': ('cache', 'type'),
            'VISIONPDF_LOG_LEVEL': ('logging', 'level'),
            'VISIONPDF_TEMP_DIR': ('temp_directory', None),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if key is None:
                    # Top-level field
                    if section == 'default_backend':
                         config_data[section] = BackendType(value)
                    else:
                         config_data[section] = value
                else:
                    # Nested field
                    if section not in config_data:
                        config_data[section] = {}

                    # Convert string values to appropriate types
                    if key == 'mode':
                        config_data[section][key] = ProcessingMode(value)
                    elif key == 'backend_type':
                        config_data[section][key] = BackendType(value)
                    elif value.lower() in ('true', 'false'):
                        config_data[section][key] = value.lower() == 'true'
                    elif value.isdigit():
                        config_data[section][key] = int(value)
                    else:
                        config_data[section][key] = value

        return cls(**config_data)

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path to save the configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary for YAML serialization
        config_dict = self.dict()

        # Convert enum values to strings
        config_dict = self._convert_enums_to_strings(config_dict)

        # Convert backend configurations
        backends_dict = {}
        for name, backend in config_dict['backends'].items():
            backends_dict[name] = {
                'backend_type': backend['backend_type'],
                'enabled': backend['enabled'],
                'config': backend['config'],
                'timeout': backend['timeout'],
                'max_retries': backend['max_retries'],
                'retry_delay': backend['retry_delay']
            }
        config_dict['backends'] = backends_dict

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _convert_enums_to_strings(self, obj: Any) -> Any:
        """Recursively convert enum values to strings for serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_strings(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        else:
            return obj

    def get_backend_config(self, backend_type: Union[str, BackendType]) -> Optional[BackendConfig]:
        """
        Get configuration for a specific backend type.

        Args:
            backend_type: Type of backend to get configuration for

        Returns:
            BackendConfig if found, None otherwise
        """
        if isinstance(backend_type, str):
            try:
                backend_type = BackendType(backend_type)
            except ValueError:
                return None

        for backend_config in self.backends.values():
            if backend_config.backend_type == backend_type:
                return backend_config
        return None

    def is_backend_enabled(self, backend_type: Union[str, BackendType]) -> bool:
        """
        Check if a backend is enabled.

        Args:
            backend_type: Type of backend to check

        Returns:
            True if backend is enabled, False otherwise
        """
        backend_config = self.get_backend_config(backend_type)
        return backend_config is not None and backend_config.enabled
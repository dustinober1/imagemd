"""
Caching utilities for VisionPDF.

This module provides intelligent caching for PDF pages, VLM responses,
and processed results to improve performance and reduce redundant processing.
"""

import hashlib
import pickle
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
import logging

from ..core.document import Document, Page, ContentElement
from ..config.settings import VisionPDFConfig
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds

    def is_access_expired(self, max_age_seconds: float) -> bool:
        """Check if entry is expired based on last access time."""
        return time.time() - self.timestamp > max_age_seconds

    def update_access(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.timestamp = time.time()


class FileCache:
    """File-based cache system for persistent storage."""

    def __init__(self, cache_dir: Path, max_size_mb: int = 1024):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, key_data: Union[str, Dict, List, Tuple]) -> str:
        """
        Generate a unique cache key from input data.

        Args:
            key_data: Data to generate key from

        Returns:
            Unique cache key string
        """
        if isinstance(key_data, str):
            key_str = key_data
        else:
            key_str = json.dumps(key_data, sort_keys=True, default=str)

        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: Union[str, Dict, List, Tuple]) -> Optional[Any]:
        """
        Get cached data.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            # Check metadata
            entry_info = self.metadata.get(cache_key)
            if entry_info:
                entry = CacheEntry(
                    data=None,  # We'll load data separately
                    timestamp=entry_info['timestamp'],
                    access_count=entry_info['access_count'],
                    size_bytes=entry_info['size_bytes'],
                    ttl_seconds=entry_info.get('ttl_seconds')
                )

                if entry.is_expired():
                    self.delete(key)
                    return None

            # Load data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Update access information
            if cache_key in self.metadata:
                self.metadata[cache_key]['access_count'] += 1
                self.metadata[cache_key]['last_access'] = time.time()
                self._save_metadata()

            logger.debug(f"Cache hit for key: {cache_key}")
            return data

        except Exception as e:
            logger.error(f"Failed to load cache entry {cache_key}: {e}")
            self.delete(key)  # Remove corrupted entry
            return None

    def set(
        self,
        key: Union[str, Dict, List, Tuple],
        data: Any,
        ttl_seconds: Optional[float] = None
    ) -> bool:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)

            # Check cache size limit
            if data_size > self.max_size_bytes:
                logger.warning(f"Cache entry too large: {data_size:,} bytes")
                return False

            # Ensure enough space
            self._ensure_space(data_size)

            # Save data
            with open(cache_file, 'wb') as f:
                f.write(serialized_data)

            # Update metadata
            self.metadata[cache_key] = {
                'timestamp': time.time(),
                'access_count': 1,
                'last_access': time.time(),
                'size_bytes': data_size,
                'ttl_seconds': ttl_seconds
            }
            self._save_metadata()

            logger.debug(f"Cached data for key: {cache_key} ({data_size:,} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache data for key {cache_key}: {e}")
            return False

    def delete(self, key: Union[str, Dict, List, Tuple]) -> bool:
        """
        Delete cached data.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found
        """
        cache_key = self._get_cache_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            cache_file.unlink(missing_ok=True)
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()

            logger.debug(f"Deleted cache entry: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete cache entry {cache_key}: {e}")
            return False

    def clear(self) -> None:
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink(missing_ok=True)

            self.metadata.clear()
            self._save_metadata()

            logger.info("Cleared all cache entries")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def _ensure_space(self, required_bytes: int) -> None:
        """
        Ensure enough space in cache by removing old entries.

        Args:
            required_bytes: Bytes required for new entry
        """
        current_size = sum(
            entry['size_bytes'] for entry in self.metadata.values()
        )

        if current_size + required_bytes <= self.max_size_bytes:
            return

        # Sort entries by last access time (LRU)
        entries_by_access = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['last_access']
        )

        # Remove old entries until enough space
        for cache_key, entry_info in entries_by_access:
            self.delete(cache_key)
            current_size -= entry_info['size_bytes']

            if current_size + required_bytes <= self.max_size_bytes:
                break

        logger.info(f"Cleaned cache to free space for {required_bytes:,} bytes")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self.metadata)
        total_size = sum(entry['size_bytes'] for entry in self.metadata.values())
        total_accesses = sum(entry['access_count'] for entry in self.metadata.values())

        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization_percent': (total_size / self.max_size_bytes) * 100,
            'total_accesses': total_accesses,
            'cache_dir': str(self.cache_dir)
        }


class MemoryCache:
    """In-memory cache for fast access to frequently used data."""

    def __init__(self, max_entries: int = 1000, max_age_seconds: float = 3600):
        """
        Initialize memory cache.

        Args:
            max_entries: Maximum number of entries
            max_age_seconds: Maximum age for entries
        """
        self.max_entries = max_entries
        self.max_age_seconds = max_age_seconds
        self._cache: Dict[str, CacheEntry] = {}

    def _get_cache_key(self, key_data: Union[str, Dict, List, Tuple]) -> str:
        """Generate cache key from input data."""
        if isinstance(key_data, str):
            key_str = key_data
        else:
            key_str = json.dumps(key_data, sort_keys=True, default=str)

        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, key: Union[str, Dict, List, Tuple]) -> Optional[Any]:
        """Get cached data."""
        cache_key = self._get_cache_key(key)
        entry = self._cache.get(cache_key)

        if entry is None:
            return None

        if entry.is_expired():
            del self._cache[cache_key]
            return None

        if entry.is_access_expired(self.max_age_seconds):
            del self._cache[cache_key]
            return None

        entry.update_access()
        return entry.data

    def set(
        self,
        key: Union[str, Dict, List, Tuple],
        data: Any,
        ttl_seconds: Optional[float] = None
    ) -> None:
        """Store data in cache."""
        cache_key = self._get_cache_key(key)

        # Ensure space
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        # Create entry
        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds,
            size_bytes=len(pickle.dumps(data))  # Rough size estimate
        )

        self._cache[cache_key] = entry

    def _evict_oldest(self) -> None:
        """Remove oldest entries to make space."""
        if not self._cache:
            return

        # Sort by timestamp and remove oldest
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].timestamp
        )
        del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'current_entries': len(self._cache),
            'max_entries': self.max_entries,
            'memory_usage_estimate': sum(entry.size_bytes for entry in self._cache.values())
        }


class PDFCache:
    """Specialized cache for PDF processing operations."""

    def __init__(self, config: VisionPDFConfig):
        """
        Initialize PDF cache.

        Args:
            config: VisionPDF configuration
        """
        self.config = config
        self.cache_enabled = config.cache.enabled

        if self.cache_enabled:
            cache_dir = Path(config.cache.directory)
            max_size_mb = config.cache.max_size_mb

            self.file_cache = FileCache(cache_dir, max_size_mb)
            self.memory_cache = MemoryCache(
                max_entries=config.cache.max_memory_entries,
                max_age_seconds=config.cache.ttl_seconds
            )
        else:
            self.file_cache = None
            self.memory_cache = None

        logger.info(f"PDF cache initialized (enabled: {self.cache_enabled})")

    def cache_pdf_analysis(self, pdf_path: Path, document: Document) -> None:
        """
        Cache PDF analysis results.

        Args:
            pdf_path: Path to PDF file
            document: Analyzed document object
        """
        if not self.cache_enabled:
            return

        cache_key = {
            'type': 'pdf_analysis',
            'path': str(pdf_path),
            'mtime': pdf_path.stat().st_mtime,
            'config': self.config.cache.get_config_hash()
        }

        # Cache in both memory and file
        self.memory_cache.set(cache_key, document, ttl_seconds=3600)
        self.file_cache.set(cache_key, document, ttl_seconds=86400)  # 24 hours

    def get_cached_analysis(self, pdf_path: Path) -> Optional[Document]:
        """
        Get cached PDF analysis.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Cached document or None
        """
        if not self.cache_enabled:
            return None

        cache_key = {
            'type': 'pdf_analysis',
            'path': str(pdf_path),
            'mtime': pdf_path.stat().st_mtime,
            'config': self.config.cache.get_config_hash()
        }

        # Try memory cache first
        document = self.memory_cache.get(cache_key)
        if document is not None:
            return document

        # Try file cache
        document = self.file_cache.get(cache_key)
        if document is not None:
            # Store in memory for faster access
            self.memory_cache.set(cache_key, document, ttl_seconds=3600)
            return document

        return None

    def cache_vlm_response(
        self,
        page_hash: str,
        backend_config: Dict[str, Any],
        response: str
    ) -> None:
        """
        Cache VLM response for a page.

        Args:
            page_hash: Hash of page content
            backend_config: Backend configuration
            response: VLM response
        """
        if not self.cache_enabled:
            return

        cache_key = {
            'type': 'vlm_response',
            'page_hash': page_hash,
            'backend_config': backend_config
        }

        # Cache with longer TTL for VLM responses (they're expensive)
        self.file_cache.set(cache_key, response, ttl_seconds=604800)  # 7 days

    def get_cached_vlm_response(
        self,
        page_hash: str,
        backend_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get cached VLM response.

        Args:
            page_hash: Hash of page content
            backend_config: Backend configuration

        Returns:
            Cached response or None
        """
        if not self.cache_enabled:
            return None

        cache_key = {
            'type': 'vlm_response',
            'page_hash': page_hash,
            'backend_config': backend_config
        }

        return self.file_cache.get(cache_key)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            return {'enabled': False}

        stats = {
            'enabled': True,
            'memory_cache': self.memory_cache.get_stats(),
            'file_cache': self.file_cache.get_stats()
        }

        return stats

    def clear_all(self) -> None:
        """Clear all cached data."""
        if not self.cache_enabled:
            return

        self.memory_cache.clear()
        self.file_cache.clear()
        logger.info("Cleared all PDF cache data")
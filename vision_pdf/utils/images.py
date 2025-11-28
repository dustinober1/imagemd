"""
Image processing utilities for VisionPDF.

This module provides tools for image processing, conversion, optimization,
and format handling throughout the PDF processing pipeline.
"""

import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PIL/Pillow not available - image processing features will be limited")

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """
    Image processing utilities for PDF page images.

    This class provides methods for image optimization, format conversion,
    and quality enhancement for VLM processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize image processor.

        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.max_size = self.config.get('max_image_size', 1024 * 1024)  # 1MB
        self.max_dimensions = self.config.get('max_dimensions', (2048, 2048))
        self.quality = self.config.get('quality', 85)
        self.optimize = self.config.get('optimize', True)
        self.supported_formats = ['PNG', 'JPEG', 'WEBP']

    def optimize_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: Optional[int] = None
    ) -> Path:
        """
        Optimize an image for VLM processing.

        Args:
            image_path: Path to input image
            output_path: Path for optimized output (overwrites input if None)
            max_width: Maximum width constraint
            max_height: Maximum height constraint
            quality: JPEG quality (1-100)

        Returns:
            Path to optimized image
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - returning original image")
            return Path(image_path)

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path
        else:
            output_path = Path(output_path)

        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Resize if necessary
                max_w = max_width or self.max_dimensions[0]
                max_h = max_height or self.max_dimensions[1]
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

                # Apply basic enhancements
                img = self._enhance_image(img)

                # Determine format
                output_format = img.format or 'JPEG'
                if output_format == 'PNG':
                    # For PNG, use different optimization
                    img.save(
                        output_path,
                        format='PNG',
                        optimize=self.optimize
                    )
                else:
                    # For JPEG/WEBP, use quality setting
                    q = quality or self.quality
                    img.save(
                        output_path,
                        format=output_format,
                        quality=q,
                        optimize=self.optimize
                    )

            logger.debug(f"Optimized image: {image_path} -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to optimize image {image_path}: {e}")
            # Return original image if optimization fails
            return image_path

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply basic image enhancements."""
        try:
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)

            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)

            return img

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return img

    def encode_image_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode image as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        image_path = Path(image_path)

        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to encode image {image_path} as base64: {e}")
            raise

    def decode_image_base64(self, base64_string: str, output_path: Union[str, Path]) -> Path:
        """
        Decode base64 string to image file.

        Args:
            base64_string: Base64 encoded image string
            output_path: Path to save decoded image

        Returns:
            Path to decoded image file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            image_data = base64.b64decode(base64_string)
            with open(output_path, 'wb') as f:
                f.write(image_data)

            return output_path

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise

    def get_image_info(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about an image file.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with image information
        """
        image_path = Path(image_path)
        info = {
            'path': str(image_path),
            'exists': image_path.exists(),
            'size_bytes': 0,
            'format': None,
            'mode': None,
            'width': 0,
            'height': 0
        }

        if not image_path.exists():
            return info

        try:
            # File size
            info['size_bytes'] = image_path.stat().st_size

            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    info['format'] = img.format
                    info['mode'] = img.mode
                    info['width'] = img.width
                    info['height'] = img.height

        except Exception as e:
            logger.error(f"Failed to get image info for {image_path}: {e}")

        return info

    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate that a file is a valid image.

        Args:
            image_path: Path to image file

        Returns:
            True if valid image, False otherwise
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return False

        try:
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    img.verify()  # Verify image integrity
                return True
            else:
                # Basic file extension check if PIL not available
                return image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']

        except Exception:
            return False

    def batch_optimize_images(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Path]:
        """
        Optimize multiple images in batch.

        Args:
            image_paths: List of image paths to optimize
            output_dir: Directory to save optimized images
            progress_callback: Optional progress callback

        Returns:
            List of paths to optimized images
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        optimized_paths = []

        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths))

            try:
                image_path = Path(image_path)

                # Determine output path
                if output_dir:
                    output_path = output_dir / image_path.name
                else:
                    output_path = None

                # Optimize image
                optimized_path = self.optimize_image(image_path, output_path)
                optimized_paths.append(optimized_path)

            except Exception as e:
                logger.error(f"Failed to optimize image {image_path}: {e}")
                optimized_paths.append(None)  # Keep order consistent

        return optimized_paths


class ImageFormatConverter:
    """Convert images between different formats."""

    def __init__(self):
        """Initialize format converter."""
        self.supported_conversions = {
            'PNG': ['JPEG', 'WEBP'],
            'JPEG': ['PNG', 'WEBP'],
            'WEBP': ['PNG', 'JPEG'],
            'BMP': ['PNG', 'JPEG', 'WEBP'],
            'TIFF': ['PNG', 'JPEG', 'WEBP']
        }

    def convert_image(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str,
        quality: int = 85
    ) -> Path:
        """
        Convert image to target format.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            target_format: Target format (PNG, JPEG, WEBP)
            quality: Quality for lossy formats (1-100)

        Returns:
            Path to converted image
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available for image conversion")

        input_path = Path(input_path)
        output_path = Path(output_path)
        target_format = target_format.upper()

        try:
            with Image.open(input_path) as img:
                # Handle transparency for JPEG conversion
                if target_format == 'JPEG' and img.mode in ('RGBA', 'LA'):
                    # Create white background for JPEG
                    background = Image.new('RGB', img.size, 'white')
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background

                # Determine output format-specific options
                save_kwargs = {'format': target_format}

                if target_format in ['JPEG', 'WEBP']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                elif target_format == 'PNG':
                    save_kwargs['optimize'] = True

                # Convert and save
                img.save(output_path, **save_kwargs)

            logger.debug(f"Converted {input_path} to {target_format}: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to convert image {input_path} to {target_format}: {e}")
            raise

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """
        Check if conversion is supported.

        Args:
            source_format: Source image format
            target_format: Target image format

        Returns:
            True if conversion is supported
        """
        source_format = source_format.upper()
        target_format = target_format.upper()

        return (
            source_format == target_format or
            target_format in self.supported_conversions.get(source_format, [])
        )


class ImageCache:
    """Cache for processed images to avoid reprocessing."""

    def __init__(self, cache_dir: Union[str, Path], max_size_mb: int = 100):
        """
        Initialize image cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, image_path: Path, processing_params: Dict[str, Any]) -> str:
        """Generate cache key for image and processing parameters."""
        import hashlib

        key_data = {
            'path': str(image_path),
            'mtime': image_path.stat().st_mtime if image_path.exists() else 0,
            'params': processing_params
        }

        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def get_cached_image(
        self,
        image_path: Path,
        processing_params: Dict[str, Any]
    ) -> Optional[Path]:
        """
        Get cached processed image if available.

        Args:
            image_path: Original image path
            processing_params: Processing parameters used

        Returns:
            Path to cached image or None
        """
        cache_key = self._get_cache_key(image_path, processing_params)
        cache_path = self.cache_dir / f"{cache_key}.jpg"

        if cache_path.exists():
            return cache_path

        return None

    def cache_image(
        self,
        image_path: Path,
        processed_image: Path,
        processing_params: Dict[str, Any]
    ) -> Path:
        """
        Cache a processed image.

        Args:
            image_path: Original image path
            processed_image: Path to processed image
            processing_params: Processing parameters used

        Returns:
            Path to cached image
        """
        cache_key = self._get_cache_key(image_path, processing_params)
        cache_path = self.cache_dir / f"{cache_key}.jpg"

        try:
            # Copy processed image to cache
            import shutil
            shutil.copy2(processed_image, cache_path)

            # Clean up cache if it gets too large
            self._cleanup_cache()

            return cache_path

        except Exception as e:
            logger.error(f"Failed to cache image: {e}")
            return processed_image

    def _cleanup_cache(self) -> None:
        """Clean up cache directory to maintain size limits."""
        try:
            cache_files = list(self.cache_dir.glob("*.jpg"))
            total_size = sum(f.stat().st_size for f in cache_files)

            if total_size > self.max_size_bytes:
                # Sort files by modification time (oldest first)
                cache_files.sort(key=lambda f: f.stat().st_mtime)

                # Remove oldest files until under limit
                for cache_file in cache_files:
                    cache_file.unlink()
                    total_size -= cache_file.stat().st_size

                    if total_size <= self.max_size_bytes * 0.8:  # Leave 20% headroom
                        break

        except Exception as e:
            logger.error(f"Failed to cleanup image cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached images."""
        try:
            for cache_file in self.cache_dir.glob("*.jpg"):
                cache_file.unlink()
            logger.info("Cleared image cache")

        except Exception as e:
            logger.error(f"Failed to clear image cache: {e}")
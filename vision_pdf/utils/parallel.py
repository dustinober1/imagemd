"""
Parallel processing utilities for VisionPDF.

This module provides tools for parallel processing of PDF pages,
batch processing, and resource management.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import logging

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for parallel processing operations."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    average_processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def total_time(self) -> float:
        """Get total processing time."""
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Calculate items processed per second."""
        total_time = self.total_time
        if total_time > 0:
            return self.completed_items / total_time
        return 0.0


class ParallelProcessor:
    """
    Generic parallel processor for async operations.

    This class provides a flexible framework for processing items
    in parallel with configurable concurrency and error handling.
    """

    def __init__(
        self,
        max_workers: int = 4,
        executor_type: str = "thread",  # thread, process
        timeout: Optional[float] = None,
        retry_attempts: int = 0,
        retry_delay: float = 1.0
    ):
        """
        Initialize parallel processor.

        Args:
            max_workers: Maximum number of concurrent workers
            executor_type: Type of executor to use
            timeout: Timeout for individual operations
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts
        """
        self.max_workers = max_workers
        self.executor_type = executor_type
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        # Statistics
        self.stats = ProcessingStats()

    async def process_items(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **process_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple items in parallel.

        Args:
            items: List of items to process
            process_func: Function to process each item
            progress_callback: Optional progress callback
            **process_kwargs: Additional arguments for process function

        Returns:
            List of processing results with success/failure info
        """
        self.stats = ProcessingStats()
        self.stats.total_items = len(items)
        self.stats.start_time = time.time()

        logger.info(f"Starting parallel processing of {len(items)} items with {self.max_workers} workers")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        # Create tasks for all items
        tasks = [
            self._process_item_with_semaphore(
                semaphore, item, process_func, i, progress_callback, **process_kwargs
            )
            for i, item in enumerate(items)
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'index': i,
                    'success': False,
                    'error': str(result),
                    'result': None
                })
                self.stats.failed_items += 1
            else:
                processed_results.append(result)
                if result.get('success', False):
                    self.stats.completed_items += 1
                else:
                    self.stats.failed_items += 1

        self.stats.end_time = time.time()
        self.stats.average_processing_time = (
            self.stats.total_time / max(1, self.stats.completed_items)
        )

        logger.info(
            f"Parallel processing complete: {self.stats.completed_items} success, "
            f"{self.stats.failed_items} failed ({self.stats.success_rate:.1f}% success rate)"
        )

        return processed_results

    async def _process_item_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        item: Any,
        process_func: Callable,
        index: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **process_kwargs
    ) -> Dict[str, Any]:
        """Process a single item with semaphore control."""
        async with semaphore:
            return await self._process_item_with_retry(
                item, process_func, index, progress_callback, **process_kwargs
            )

    async def _process_item_with_retry(
        self,
        item: Any,
        process_func: Callable,
        index: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **process_kwargs
    ) -> Dict[str, Any]:
        """Process a single item with retry logic."""
        last_exception = None

        for attempt in range(self.retry_attempts + 1):
            try:
                # Process the item
                if asyncio.iscoroutinefunction(process_func):
                    result = await asyncio.wait_for(
                        process_func(item, **process_kwargs),
                        timeout=self.timeout
                    )
                else:
                    # Run synchronous function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: process_func(item, **process_kwargs)
                    )

                # Update progress
                if progress_callback:
                    progress_callback(index + 1, self.stats.total_items)

                return {
                    'index': index,
                    'success': True,
                    'result': result,
                    'error': None,
                    'attempts': attempt + 1
                }

            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"Processing timeout after {self.timeout}s")
                logger.warning(f"Item {index} timed out (attempt {attempt + 1})")

            except Exception as e:
                last_exception = e
                logger.warning(f"Item {index} failed (attempt {attempt + 1}): {e}")

            # Retry delay
            if attempt < self.retry_attempts:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

        # All attempts failed
        return {
            'index': index,
            'success': False,
            'result': None,
            'error': str(last_exception),
            'attempts': self.retry_attempts + 1
        }

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        return self.stats


class BatchProcessor:
    """
    Batch processor for handling large lists of items.

    This class processes items in configurable batches to manage
    memory usage and provide better control over processing flow.
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
        timeout: Optional[float] = None
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum number of batches to process concurrently
            timeout: Timeout for each batch
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.timeout = timeout

    async def process_batches(
        self,
        items: List[Any],
        process_batch_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **process_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process items in batches.

        Args:
            items: List of items to process
            process_batch_func: Function to process each batch
            progress_callback: Optional progress callback
            **process_kwargs: Additional arguments for batch processing function

        Returns:
            List of batch processing results
        """
        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        # Create semaphore for batch concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        # Process batches
        batch_tasks = [
            self._process_batch_with_semaphore(
                semaphore, batch, process_batch_func, i, progress_callback, **process_kwargs
            )
            for i, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                # Add error result for all items in this batch
                all_results.extend([{
                    'success': False,
                    'error': str(result),
                    'result': None
                }])
            else:
                all_results.extend(result)

        return all_results

    async def _process_batch_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        batch: List[Any],
        process_batch_func: Callable,
        batch_index: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **process_kwargs
    ) -> List[Dict[str, Any]]:
        """Process a single batch with semaphore control."""
        async with semaphore:
            try:
                # Process the batch
                if asyncio.iscoroutinefunction(process_batch_func):
                    results = await asyncio.wait_for(
                        process_batch_func(batch, **process_kwargs),
                        timeout=self.timeout
                    )
                else:
                    # Run synchronous function in thread pool
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: process_batch_func(batch, **process_kwargs)
                    )

                # Update progress
                if progress_callback:
                    progress_callback(len(batch), len(batch))

                return results

            except Exception as e:
                logger.error(f"Batch {batch_index} failed: {e}")
                # Return error results for all items in batch
                return [{
                    'success': False,
                    'error': str(e),
                    'result': None
                } for _ in batch]


class ResourceMonitor:
    """
    Monitor system resources during processing.

    This class provides tools to monitor CPU, memory, and disk usage
    and optionally throttle processing based on resource availability.
    """

    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 80.0,
        check_interval: float = 5.0
    ):
        """
        Initialize resource monitor.

        Args:
            max_cpu_percent: Maximum CPU usage percentage before throttling
            max_memory_percent: Maximum memory usage percentage before throttling
            check_interval: Interval between resource checks
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self._monitoring = False
        self._stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'throttled': False
        }

    async def start_monitoring(self):
        """Start resource monitoring in background."""
        self._monitoring = True
        asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False

    async def _monitor_loop(self):
        """Background monitoring loop."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
            return

        while self._monitoring:
            try:
                # Get resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                self._stats.update({
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'disk_usage': disk.percent,
                    'throttled': (
                        cpu_percent > self.max_cpu_percent or
                        memory.percent > self.max_memory_percent
                    )
                })

                if self._stats['throttled']:
                    logger.warning(
                        f"High resource usage - CPU: {cpu_percent:.1f}%, "
                        f"Memory: {memory.percent:.1f}%"
                    )

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    def get_stats(self) -> Dict[str, float]:
        """Get current resource statistics."""
        return self._stats.copy()

    def should_throttle(self) -> bool:
        """Check if processing should be throttled due to resource constraints."""
        return self._stats.get('throttled', False)


class ProgressTracker:
    """Track and report progress of long-running operations."""

    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.completed_items = 0
        self.description = description
        self.start_time = time.time()

    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.completed_items = min(self.completed_items + increment, self.total_items)

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.completed_items / self.total_items) * 100

        # Estimate remaining time
        if self.completed_items > 0:
            estimated_total_time = elapsed_time * (self.total_items / self.completed_items)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0.0

        return {
            'completed': self.completed_items,
            'total': self.total_items,
            'percentage': progress_percent,
            'elapsed_time': elapsed_time,
            'estimated_remaining_time': remaining_time,
            'description': self.description
        }
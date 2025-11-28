"""
Performance monitoring and optimization utilities for VisionPDF.

This module provides performance monitoring, profiling, and optimization
tools to track and improve processing performance.
"""

import time
import threading
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import json
import logging
from contextlib import contextmanager, asynccontextmanager

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    operation_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    errors: int = 0
    success_rate: float = 1.0

    def update(self, duration: float, success: bool = True):
        """Update metrics with new operation data."""
        self.total_calls += 1
        self.recent_times.append(duration)

        if success:
            self.total_time += duration
            self.min_time = min(self.min_time, duration)
            self.max_time = max(self.max_time, duration)
            self.avg_time = self.total_time / self.total_calls
        else:
            self.errors += 1

        # Calculate success rate
        self.success_rate = (self.total_calls - self.errors) / self.total_calls

    def get_recent_avg(self, count: int = 10) -> float:
        """Get average of recent times."""
        if not self.recent_times:
            return 0.0
        recent = list(self.recent_times)[-count:]
        return sum(recent) / len(recent)


class PerformanceMonitor:
    """Performance monitoring system."""

    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor.

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self._metrics: Dict[str, OperationMetrics] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()

    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record performance data for an operation."""
        if not self.enabled:
            return

        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = OperationMetrics(operation_name)

            self._metrics[operation_name].update(duration, success)

    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation performance."""
        if not self.enabled:
            yield
            return

        start_time = time.time()
        success = True
        try:
            yield
        except Exception as e:
            success = False
            logger.warning(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation(operation_name, duration, success)

    @asynccontextmanager
    async def measure_async(self, operation_name: str):
        """Async context manager to measure operation performance."""
        if not self.enabled:
            yield
            return

        start_time = time.time()
        success = True
        try:
            yield
        except Exception as e:
            success = False
            logger.warning(f"Async operation {operation_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.record_operation(f"{operation_name}_async", duration, success)

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics."""
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    return {operation_name: self._metrics_to_dict(self._metrics[operation_name])}
                return {}
            else:
                return {name: self._metrics_to_dict(metrics) for name, metrics in self._metrics.items()}

    def _metrics_to_dict(self, metrics: OperationMetrics) -> Dict[str, Any]:
        """Convert OperationMetrics to dictionary."""
        return {
            'operation_name': metrics.operation_name,
            'total_calls': metrics.total_calls,
            'total_time': metrics.total_time,
            'min_time': metrics.min_time,
            'max_time': metrics.max_time,
            'avg_time': metrics.avg_time,
            'recent_avg_time': metrics.get_recent_avg(),
            'errors': metrics.errors,
            'success_rate': metrics.success_rate
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            total_operations = sum(m.total_calls for m in self._metrics.values())
            total_errors = sum(m.errors for m in self._metrics.values())
            overall_success_rate = (total_operations - total_errors) / total_operations if total_operations > 0 else 1.0

            # Find slowest operations
            slowest_ops = sorted(
                [(name, metrics.avg_time) for name, metrics in self._metrics.items() if metrics.total_calls > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Find most frequent operations
            frequent_ops = sorted(
                [(name, metrics.total_calls) for name, metrics in self._metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                'monitoring_enabled': self.enabled,
                'uptime_seconds': time.time() - self._start_time,
                'total_operations': total_operations,
                'total_errors': total_errors,
                'overall_success_rate': overall_success_rate,
                'unique_operations': len(self._metrics),
                'slowest_operations': slowest_ops,
                'most_frequent_operations': frequent_ops
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._start_time = time.time()

    def export_metrics(self, file_path: Union[str, Path]):
        """Export metrics to JSON file."""
        file_path = Path(file_path)
        data = {
            'timestamp': time.time(),
            'summary': self.get_summary(),
            'detailed_metrics': self.get_metrics()
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported performance metrics to {file_path}")


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self, enabled: bool = True):
        """
        Initialize resource monitor.

        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self._metrics = defaultdict(list)
        self._lock = threading.RLock()

    def record_memory_usage(self, operation: str, memory_mb: float):
        """Record memory usage."""
        if not self.enabled:
            return

        with self._lock:
            self._metrics[f"{operation}_memory"].append({
                'timestamp': time.time(),
                'memory_mb': memory_mb
            })

    def record_cpu_usage(self, operation: str, cpu_percent: float):
        """Record CPU usage."""
        if not self.enabled:
            return

        with self._lock:
            self._metrics[f"{operation}_cpu"].append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent
            })

    @contextmanager
    def monitor_memory(self, operation: str):
        """Monitor memory usage during operation."""
        if not self.enabled:
            yield
            return

        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            yield
            return

        try:
            yield
        finally:
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = max(start_memory, end_memory)
            self.record_memory_usage(operation, peak_memory)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        with self._lock:
            summary = {}
            for key, values in self._metrics.items():
                if not values:
                    continue

                if key.endswith('_memory'):
                    memory_values = [v['memory_mb'] for v in values]
                    summary[key] = {
                        'avg_mb': sum(memory_values) / len(memory_values),
                        'max_mb': max(memory_values),
                        'min_mb': min(memory_values),
                        'samples': len(memory_values)
                    }
                elif key.endswith('_cpu'):
                    cpu_values = [v['cpu_percent'] for v in values]
                    summary[key] = {
                        'avg_percent': sum(cpu_values) / len(cpu_values),
                        'max_percent': max(cpu_values),
                        'min_percent': min(cpu_values),
                        'samples': len(cpu_values)
                    }

            return summary


class BatchProcessor:
    """Optimized batch processing for multiple operations."""

    def __init__(self, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of concurrent workers
            batch_size: Default batch size
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items in batches with concurrency control.

        Args:
            items: Items to process
            process_func: Async function to process each item
            batch_size: Size of each batch
            progress_callback: Optional progress callback

        Returns:
            List of processed results
        """
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        total_items = len(items)

        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, total_items, batch_size)
        ]

        async def process_batch_with_semaphore(batch):
            async with self.semaphore:
                batch_results = []
                for item in batch:
                    try:
                        result = await process_func(item)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
                        batch_results.append(None)
                return batch_results

        # Process all batches concurrently
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(len(results), total_items)

        return results

    def optimize_batch_size(self, items: List[Any], sample_size: int = 10) -> int:
        """
        Determine optimal batch size based on system characteristics.

        Args:
            items: Items to process
            sample_size: Size of sample to test with

        Returns:
            Optimal batch size
        """
        if len(items) <= self.batch_size:
            return len(items)

        # Simple heuristic based on item count and workers
        optimal_size = max(1, min(len(items) // self.max_workers, self.batch_size * 2))
        return optimal_size


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor(
            enabled=config.get('monitoring_enabled', True)
        )
        self.resource_monitor = ResourceMonitor(
            enabled=config.get('resource_monitoring_enabled', True)
        )
        self.batch_processor = BatchProcessor(
            max_workers=config.get('max_workers', 4),
            batch_size=config.get('batch_size', 10)
        )

        # Optimization settings
        self.cache_enabled = config.get('cache_enabled', True)
        self.parallel_processing = config.get('parallel_processing', True)
        self.optimization_level = config.get('optimization_level', 'balanced')  # conservative, balanced, aggressive

    @contextmanager
    def measure_operation(self, operation_name: str):
        """Measure operation performance."""
        with self.performance_monitor.measure(operation_name):
            with self.resource_monitor.monitor_memory(operation_name):
                yield

    @asynccontextmanager
    async def measure_operation_async(self, operation_name: str):
        """Measure async operation performance."""
        async with self.performance_monitor.measure_async(operation_name):
            # Note: Memory monitoring in async context requires additional handling
            yield

    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        metrics = self.performance_monitor.get_metrics()

        # Analyze slow operations
        for op_name, op_metrics in metrics.items():
            if op_metrics['avg_time'] > 5.0:  # Operations taking > 5 seconds
                recommendations.append(
                    f"Operation '{op_name}' is slow (avg: {op_metrics['avg_time']:.2f}s). "
                    "Consider caching results or optimizing the implementation."
                )

            if op_metrics['success_rate'] < 0.95:
                recommendations.append(
                    f"Operation '{op_name}' has low success rate ({op_metrics['success_rate']:.1%}). "
                    "Review error handling and retry logic."
                )

        # Resource-based recommendations
        resource_summary = self.resource_monitor.get_resource_summary()
        for key, summary in resource_summary.items():
            if key.endswith('_memory') and summary['max_mb'] > 1000:  # > 1GB
                recommendations.append(
                    f"High memory usage in {key} (max: {summary['max_mb']:.1f}MB). "
                    "Consider implementing streaming or chunked processing."
                )

            if key.endswith('_cpu') and summary['max_percent'] > 80:
                recommendations.append(
                    f"High CPU usage in {key} (max: {summary['max_percent']:.1f}%). "
                    "Consider reducing concurrency or optimizing algorithms."
                )

        if not recommendations:
            recommendations.append("Performance looks good! No major optimizations needed.")

        return recommendations

    def generate_performance_report(self, output_path: Optional[Union[str, Path]] = None):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.time(),
            'performance_summary': self.performance_monitor.get_summary(),
            'detailed_metrics': self.performance_monitor.get_metrics(),
            'resource_summary': self.resource_monitor.get_resource_summary(),
            'optimization_recommendations': self.get_optimization_recommendations(),
            'configuration': {
                'monitoring_enabled': self.performance_monitor.enabled,
                'cache_enabled': self.cache_enabled,
                'parallel_processing': self.parallel_processing,
                'max_workers': self.batch_processor.max_workers,
                'batch_size': self.batch_processor.batch_size
            }
        }

        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance report saved to {output_path}")

        return report


# Global performance monitor instance
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def measure_performance(operation_name: str):
    """Decorator to measure function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.measure(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_performance_async(operation_name: str):
    """Decorator to measure async function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            async with monitor.measure_async(operation_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator
"""
Tests for performance optimization and caching system.

This module tests the performance monitoring, resource tracking,
batch processing, and optimization recommendations.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from vision_pdf.utils.performance import (
    PerformanceMonitor, PerformanceMetric, OperationMetrics,
    ResourceMonitor, BatchProcessor, PerformanceOptimizer,
    measure_performance, measure_performance_async, get_performance_monitor
)


class TestPerformanceMetric:
    """Test performance metric data structure."""

    def test_metric_creation(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(
            name="test_operation",
            value=1.5,
            unit="seconds",
            timestamp=time.time()
        )
        assert metric.name == "test_operation"
        assert metric.value == 1.5
        assert metric.unit == "seconds"
        assert isinstance(metric.timestamp, float)


class TestOperationMetrics:
    """Test operation metrics tracking."""

    def test_metrics_initialization(self):
        """Test operation metrics initialization."""
        metrics = OperationMetrics("test_operation")
        assert metrics.operation_name == "test_operation"
        assert metrics.total_calls == 0
        assert metrics.total_time == 0.0
        assert metrics.min_time == float('inf')
        assert metrics.max_time == 0.0

    def test_metrics_update(self):
        """Test updating metrics with operation data."""
        metrics = OperationMetrics("test_operation")

        # Update with successful operation
        metrics.update(1.5, True)
        assert metrics.total_calls == 1
        assert metrics.total_time == 1.5
        assert metrics.min_time == 1.5
        assert metrics.max_time == 1.5
        assert metrics.avg_time == 1.5
        assert metrics.success_rate == 1.0

        # Update with another successful operation
        metrics.update(2.5, True)
        assert metrics.total_calls == 2
        assert metrics.total_time == 4.0
        assert metrics.min_time == 1.5
        assert metrics.max_time == 2.5
        assert metrics.avg_time == 2.0

        # Update with failed operation
        metrics.update(1.0, False)
        assert metrics.total_calls == 3
        assert metrics.total_time == 4.0  # Failed operations don't affect time
        assert metrics.errors == 1
        assert metrics.success_rate == 2.0 / 3.0

    def test_recent_average(self):
        """Test calculating recent average."""
        metrics = OperationMetrics("test_operation")

        # No recent times
        assert metrics.get_recent_avg() == 0.0

        # Add some times
        metrics.recent_times.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert metrics.get_recent_avg() == 3.0
        assert metrics.get_recent_avg(count=3) == 4.0  # Average of last 3


class TestPerformanceMonitor:
    """Test performance monitoring system."""

    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(enabled=True)
        assert monitor.enabled is True
        assert len(monitor._metrics) == 0
        assert isinstance(monitor._start_time, float)

        disabled_monitor = PerformanceMonitor(enabled=False)
        assert disabled_monitor.enabled is False

    def test_record_operation(self):
        """Test recording operation performance."""
        monitor = PerformanceMonitor(enabled=True)

        # Record successful operation
        monitor.record_operation("test_op", 1.5, True)
        metrics = monitor._metrics["test_op"]
        assert metrics.total_calls == 1
        assert metrics.total_time == 1.5

        # Record failed operation
        monitor.record_operation("test_op", 2.0, False)
        assert metrics.total_calls == 2
        assert metrics.errors == 1

        # Test disabled monitor
        disabled_monitor = PerformanceMonitor(enabled=False)
        disabled_monitor.record_operation("test_op", 1.5, True)
        assert len(disabled_monitor._metrics) == 0

    def test_measure_context_manager(self):
        """Test measurement context manager."""
        monitor = PerformanceMonitor(enabled=True)

        # Successful operation
        with monitor.measure("test_op"):
            time.sleep(0.1)

        metrics = monitor._metrics["test_op"]
        assert metrics.total_calls == 1
        assert metrics.errors == 0
        assert metrics.avg_time > 0.1

        # Failed operation
        with pytest.raises(ValueError):
            with monitor.measure("test_op"):
                time.sleep(0.05)
                raise ValueError("Test error")

        metrics = monitor._metrics["test_op"]
        assert metrics.total_calls == 2
        assert metrics.errors == 1

    @pytest.mark.asyncio
    async def test_measure_async_context_manager(self):
        """Test async measurement context manager."""
        monitor = PerformanceMonitor(enabled=True)

        async def slow_operation():
            await asyncio.sleep(0.1)
            return "result"

        # Successful async operation
        async with monitor.measure_async("async_test_op"):
            result = await slow_operation()
            assert result == "result"

        metrics = monitor._metrics["async_test_op_async"]
        assert metrics.total_calls == 1
        assert metrics.errors == 0

    def test_get_metrics(self):
        """Test getting performance metrics."""
        monitor = PerformanceMonitor(enabled=True)

        # Record some operations
        monitor.record_operation("op1", 1.0, True)
        monitor.record_operation("op1", 2.0, True)
        monitor.record_operation("op2", 3.0, False)

        # Get all metrics
        all_metrics = monitor.get_metrics()
        assert "op1" in all_metrics
        assert "op2" in all_metrics
        assert all_metrics["op1"]["total_calls"] == 2
        assert all_metrics["op2"]["total_calls"] == 1

        # Get specific operation metrics
        op1_metrics = monitor.get_metrics("op1")
        assert "op1" in op1_metrics
        assert "op2" not in op1_metrics

    def test_get_summary(self):
        """Test performance summary."""
        monitor = PerformanceMonitor(enabled=True)

        # Record some operations
        monitor.record_operation("op1", 1.0, True)
        monitor.record_operation("op2", 2.0, False)

        summary = monitor.get_summary()
        assert summary["monitoring_enabled"] is True
        assert summary["total_operations"] == 2
        assert summary["total_errors"] == 1
        assert summary["overall_success_rate"] == 0.5
        assert summary["unique_operations"] == 2
        assert isinstance(summary["uptime_seconds"], float)

    def test_reset(self):
        """Test resetting metrics."""
        monitor = PerformanceMonitor(enabled=True)

        # Record some operations
        monitor.record_operation("test_op", 1.0, True)
        assert len(monitor._metrics) == 1

        # Reset
        monitor.reset()
        assert len(monitor._metrics) == 0
        assert monitor._start_time > time.time() - 1.0  # Should be recent

    def test_export_metrics(self):
        """Test exporting metrics to file."""
        monitor = PerformanceMonitor(enabled=True)

        # Record some operations
        monitor.record_operation("test_op", 1.0, True)

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            monitor.export_metrics(temp_path)
            assert Path(temp_path).exists()

            # Verify content
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "summary" in data
            assert "detailed_metrics" in data
            assert "test_op" in data["detailed_metrics"]

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestResourceMonitor:
    """Test resource monitoring."""

    @patch('vision_pdf.utils.performance.psutil')
    def test_record_memory_usage(self, mock_psutil):
        """Test recording memory usage."""
        monitor = ResourceMonitor(enabled=True)

        monitor.record_memory_usage("test_op", 256.5)
        assert "test_op_memory" in monitor._metrics
        assert len(monitor._metrics["test_op_memory"]) == 1
        assert monitor._metrics["test_op_memory"][0]["memory_mb"] == 256.5

        # Test disabled monitor
        disabled_monitor = ResourceMonitor(enabled=False)
        disabled_monitor.record_memory_usage("test_op", 256.5)
        assert len(disabled_monitor._metrics) == 0

    def test_record_cpu_usage(self):
        """Test recording CPU usage."""
        monitor = ResourceMonitor(enabled=True)

        monitor.record_cpu_usage("test_op", 75.2)
        assert "test_op_cpu" in monitor._metrics
        assert len(monitor._metrics["test_op_cpu"]) == 1
        assert monitor._metrics["test_op_cpu"][0]["cpu_percent"] == 75.2

    @patch('vision_pdf.utils.performance.psutil')
    def test_monitor_memory_context(self, mock_psutil):
        """Test memory monitoring context manager."""
        # Mock psutil
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024 * 100)  # 100MB
        mock_psutil.Process.return_value = mock_process

        monitor = ResourceMonitor(enabled=True)

        with monitor.monitor_memory("test_op"):
            # Simulate some operation
            time.sleep(0.01)

        assert "test_op_memory" in monitor._metrics
        memory_record = monitor._metrics["test_op_memory"][0]
        assert memory_record["memory_mb"] == 100.0

    def test_get_resource_summary(self):
        """Test getting resource summary."""
        monitor = ResourceMonitor(enabled=True)

        # Add some metrics
        monitor.record_memory_usage("op1", 256.0)
        monitor.record_memory_usage("op1", 512.0)
        monitor.record_cpu_usage("op2", 75.0)

        summary = monitor.get_resource_summary()
        assert "op1_memory" in summary
        assert "op2_cpu" in summary

        memory_summary = summary["op1_memory"]
        assert memory_summary["avg_mb"] == 384.0
        assert memory_summary["max_mb"] == 512.0
        assert memory_summary["min_mb"] == 256.0
        assert memory_summary["samples"] == 2


class TestBatchProcessor:
    """Test batch processing system."""

    def test_initialization(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(max_workers=8, batch_size=20)
        assert processor.max_workers == 8
        assert processor.batch_size == 20
        assert processor.semaphore._value == 8

    @pytest.mark.asyncio
    async def test_process_batch(self):
        """Test batch processing."""
        processor = BatchProcessor(max_workers=2, batch_size=2)

        async def process_item(item):
            """Simple async processing function."""
            await asyncio.sleep(0.01)
            return item * 2

        # Test data
        items = [1, 2, 3, 4, 5]

        # Process batch
        results = await processor.process_batch(items, process_item)

        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_process_batch_with_callback(self):
        """Test batch processing with progress callback."""
        processor = BatchProcessor(max_workers=2, batch_size=2)

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        async def process_item(item):
            await asyncio.sleep(0.01)
            return item

        items = [1, 2, 3, 4]

        await processor.process_batch(items, process_item, progress_callback=progress_callback)

        # Check progress was reported
        assert len(progress_calls) > 0
        assert progress_calls[-1] == (4, 4)  # Final call should show completion

    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        processor = BatchProcessor(max_workers=4, batch_size=10)

        # Small list
        small_items = list(range(5))
        optimal = processor.optimize_batch_size(small_items)
        assert optimal == 5

        # Large list
        large_items = list(range(100))
        optimal = processor.optimize_batch_size(large_items)
        assert optimal > 5  # Should be larger than default
        assert optimal <= 100


class TestPerformanceOptimizer:
    """Test performance optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        config = {
            'monitoring_enabled': True,
            'resource_monitoring_enabled': True,
            'max_workers': 8,
            'batch_size': 20,
            'optimization_level': 'aggressive'
        }

        optimizer = PerformanceOptimizer(config)

        assert optimizer.performance_monitor.enabled is True
        assert optimizer.resource_monitor.enabled is True
        assert optimizer.batch_processor.max_workers == 8
        assert optimizer.batch_processor.batch_size == 20

    @pytest.mark.asyncio
    async def test_measure_operation_async(self):
        """Test async operation measurement."""
        config = {'monitoring_enabled': True}
        optimizer = PerformanceOptimizer(config)

        async def test_operation():
            await asyncio.sleep(0.01)
            return "result"

        async with optimizer.measure_operation_async("test_async_op"):
            result = await test_operation()
            assert result == "result"

        metrics = optimizer.performance_monitor.get_metrics()
        assert "test_async_op_async" in metrics

    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        config = {'monitoring_enabled': True}
        optimizer = PerformanceOptimizer(config)

        # Add some slow operations
        optimizer.performance_monitor.record_operation("slow_op", 6.0, True)
        optimizer.performance_monitor.record_operation("error_op", 1.0, False)
        optimizer.performance_monitor.record_operation("error_op", 1.0, False)

        recommendations = optimizer.get_optimization_recommendations()
        assert len(recommendations) > 0

        # Should recommend optimization for slow operation
        slow_rec = [r for r in recommendations if "slow_op" in r]
        assert len(slow_rec) > 0

        # Should recommend error handling improvement
        error_rec = [r for r in recommendations if "error_op" in r]
        assert len(error_rec) > 0

    def test_generate_performance_report(self):
        """Test performance report generation."""
        config = {'monitoring_enabled': True}
        optimizer = PerformanceOptimizer(config)

        # Add some data
        optimizer.performance_monitor.record_operation("test_op", 1.0, True)

        # Generate report
        report = optimizer.generate_performance_report()

        assert "timestamp" in report
        assert "performance_summary" in report
        assert "detailed_metrics" in report
        assert "resource_summary" in report
        assert "optimization_recommendations" in report
        assert "configuration" in report

        # Test saving to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            optimizer.generate_performance_report(temp_path)
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPerformanceDecorators:
    """Test performance measurement decorators."""

    def test_measure_performance_decorator(self):
        """Test synchronous performance measurement decorator."""
        @measure_performance("decorated_function")
        def test_function(x, y):
            time.sleep(0.01)
            return x + y

        result = test_function(2, 3)
        assert result == 5

        # Check that metrics were recorded
        monitor = get_performance_monitor()
        metrics = monitor.get_metrics()
        assert "decorated_function" in metrics

    @pytest.mark.asyncio
    async def test_measure_performance_async_decorator(self):
        """Test async performance measurement decorator."""
        @measure_performance_async("async_decorated_function")
        async def async_test_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_test_function(5)
        assert result == 10

        # Check that metrics were recorded
        monitor = get_performance_monitor()
        metrics = monitor.get_metrics()
        assert "async_decorated_function_async" in metrics

    def test_global_performance_monitor(self):
        """Test global performance monitor."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        # Should return the same instance
        assert monitor1 is monitor2


if __name__ == "__main__":
    pytest.main([__file__])
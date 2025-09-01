"""Timing and memory profiling utilities for latency benchmarking.

Provides context managers and decorators for measuring wall-clock time
and peak memory usage across pipeline stages. Results feed into
latency benchmarks (P50/P95/P99) and model footprint reports.
"""

import functools
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, List, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TimingResult:
    """Container for timing and memory measurements.

    Attributes:
        name: Label for this measurement.
        elapsed_sec: Total wall-clock time in seconds.
        peak_memory_mb: Peak memory usage in megabytes.
        samples: Number of samples processed (for throughput).
        throughput: Items per second (samples / elapsed).
    """

    name: str
    elapsed_sec: float
    peak_memory_mb: float = 0.0
    samples: Optional[int] = None
    throughput: Optional[float] = None

    def __post_init__(self) -> None:
        if self.samples is not None and self.elapsed_sec > 0:
            self.throughput = self.samples / self.elapsed_sec

    def __str__(self) -> str:
        parts = [
            f"{self.name}: {self.elapsed_sec:.3f}s",
            f"peak_mem={self.peak_memory_mb:.1f}MB",
        ]
        if self.throughput is not None:
            parts.append(f"throughput={self.throughput:.0f}/s")
        return " | ".join(parts)


@contextmanager
def timer(name: str, samples: Optional[int] = None) -> Generator[TimingResult, None, None]:
    """Context manager for timing code blocks.

    Measures elapsed wall-clock time. Logs the result automatically on exit.

    Args:
        name: Descriptive label for the timed block.
        samples: Optional count of items processed (enables throughput).

    Yields:
        TimingResult that will be populated on context exit.
    """
    result = TimingResult(name=name, elapsed_sec=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result.elapsed_sec = elapsed
        if samples is not None:
            result.samples = samples
            result.throughput = samples / elapsed if elapsed > 0 else 0.0

        log.info(str(result))


@dataclass
class LatencyBenchmark:
    """Accumulates timing results for P50/P95/P99 latency statistics.

    Attributes:
        name: Benchmark label.
        measurements: List of individual timing measurements in seconds.
    """

    name: str
    measurements: List[float] = field(default_factory=list)

    def record(self, elapsed_sec: float) -> None:
        """Record a single timing observation.

        Args:
            elapsed_sec: Elapsed time in seconds for one inference call.
        """
        self.measurements.append(elapsed_sec)

    def summary(self) -> dict:
        """Compute latency percentile statistics.

        Returns:
            Dictionary with p50, p95, p99, mean, std, count.

        Raises:
            ValueError: If no measurements have been recorded.
        """
        import numpy as np

        if not self.measurements:
            raise ValueError("No timing measurements recorded")

        arr = np.array(self.measurements) * 1000  # convert to ms
        return {
            "name": self.name,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "count": len(arr),
            "total_sec": float(np.sum(self.measurements)),
        }


def timeit(func: Callable) -> Callable:
    """Decorator to time function calls and log results.

    Args:
        func: The function to wrap with timing.

    Returns:
        Wrapped function that logs execution time on each call.

    Example:
        >>> @timeit
        ... def train_model(data):
        ...     pass
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap func call with timing and debug logging."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.debug(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result

    return wrapper

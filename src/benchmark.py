"""
Benchmarking Framework for Sorting Algorithm Performance Analysis

This module provides high-precision timing and memory profiling utilities
for empirical complexity analysis. The framework ensures experimental
integrity through data isolation and accurate resource measurement.

Measurement Techniques:
    - Timing: time.perf_counter() for nanosecond precision
    - Memory: tracemalloc for peak allocation tracking
    - Isolation: Data copying prevents cross-experiment contamination

Author: Swornim Shrestha
Date: January 2026
References:
    - SortBenchmarker: https://github.com/sanaynesargi/SortBenchmarker
    - mexwell Kaggle: https://www.kaggle.com/code/mexwell/exploring-sorting-algorithms-in-python
"""

import time
import tracemalloc
from typing import Callable, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """
    Container for individual benchmark measurement results.
    
    Attributes:
        algorithm_name: Name of the sorting algorithm tested
        distribution: Type of input distribution (Random, Sorted, Reverse)
        input_size: Number of elements in the test array
        execution_time: Time taken in seconds (float)
        peak_memory_kb: Peak memory usage in kilobytes (float)
        is_correct: Verification flag for sort correctness
    """
    algorithm_name: str
    distribution: str
    input_size: int
    execution_time: float  # seconds
    peak_memory_kb: float  # kilobytes
    is_correct: bool


def measure_time(algorithm: Callable, data: List[int]) -> Tuple[float, List[int]]:
    """
    Measure execution time of a sorting algorithm with high precision.
    
    Uses time.perf_counter() which provides the highest available resolution
    for performance measurement on the current platform. This is preferred
    over time.time() for benchmarking as it's monotonic and not affected
    by system clock adjustments.
    
    Args:
        algorithm: Sorting function to benchmark
        data: Input array (will be copied to prevent mutation)
        
    Returns:
        Tuple of (execution_time_seconds, sorted_result)
        
    Example:
        >>> def bubble_sort(arr): return sorted(arr)
        >>> time_taken, result = measure_time(bubble_sort, [3, 1, 2])
        >>> print(f"Time: {time_taken:.6f}s, Result: {result}")
        Time: 0.000012s, Result: [1, 2, 3]
        
    Note:
        Data is copied before sorting to ensure input immutability and
        prevent timing contamination from in-place modifications.
    """
    # Create isolated copy to prevent cross-experiment contamination
    data_copy = data.copy()
    
    # High-precision timing
    start_time = time.perf_counter()
    result = algorithm(data_copy)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    return execution_time, result


def measure_memory(algorithm: Callable, data: List[int]) -> Tuple[float, List[int]]:
    """
    Measure peak memory usage during algorithm execution.
    
    Uses Python's tracemalloc module to track memory allocations.
    This provides insight into space complexity beyond theoretical
    analysis, capturing actual runtime memory footprint including
    recursion stack and auxiliary structures.
    
    Args:
        algorithm: Sorting function to benchmark
        data: Input array (will be copied to prevent mutation)
        
    Returns:
        Tuple of (peak_memory_kb, sorted_result)
        
    Example:
        >>> def merge_sort(arr): return sorted(arr)
        >>> mem_used, result = measure_memory(merge_sort, [5, 2, 8, 1])
        >>> print(f"Memory: {mem_used:.2f} KB")
        Memory: 1.24 KB
        
    Note:
        Memory measurements are relative and include Python interpreter
        overhead. Focus on comparative trends rather than absolute values.
        
    Limitations:
        - tracemalloc has ~10% performance overhead
        - Peak memory may include garbage not yet collected
        - Results vary based on Python's memory allocation strategy
    """
    # Create isolated copy
    data_copy = data.copy()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Execute algorithm
    result = algorithm(data_copy)
    
    # Capture peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert bytes to kilobytes for readability
    peak_memory_kb = peak / 1024.0
    
    return peak_memory_kb, result


def verify_correctness(result: List[int], original: List[int]) -> bool:
    """
    Verify that sorting algorithm produced correct output.
    
    Checks if the result is:
        1. Properly sorted in ascending order
        2. Contains the same elements as the original input
    
    Args:
        result: Output from sorting algorithm
        original: Original unsorted input
        
    Returns:
        True if sort is correct, False otherwise
        
    Example:
        >>> verify_correctness([1, 2, 3], [3, 1, 2])
        True
        >>> verify_correctness([1, 2], [3, 1, 2])
        False
    """
    # Check if sorted
    if result != sorted(result):
        return False
    
    # Check if contains same elements (handles duplicates)
    if sorted(result) != sorted(original):
        return False
    
    return True


def benchmark_algorithm(
    algorithm: Callable,
    algorithm_name: str,
    data: List[int],
    distribution_name: str
) -> BenchmarkResult:
    """
    Perform comprehensive benchmark of a single algorithm on given data.
    
    Conducts both time and memory measurements in separate runs to avoid
    instrumentation interference. Each measurement uses fresh data copies
    to ensure experimental independence.
    
    Args:
        algorithm: Sorting function to test
        algorithm_name: Human-readable algorithm identifier
        data: Input dataset
        distribution_name: Data distribution type (Random/Sorted/Reverse)
        
    Returns:
        BenchmarkResult object containing all measurements and metadata
        
    Example:
        >>> from algorithms import bubble_sort
        >>> data = [5, 2, 8, 1, 9]
        >>> result = benchmark_algorithm(
        ...     bubble_sort, "Bubble Sort", data, "Random"
        ... )
        >>> print(f"{result.algorithm_name}: {result.execution_time:.6f}s")
        Bubble Sort: 0.000023s
        
    Note:
        Time and memory are measured in separate runs because tracemalloc
        adds ~10% overhead that would skew timing results.
    """
    input_size = len(data)
    
    # Measure execution time
    execution_time, time_result = measure_time(algorithm, data)
    
    # Measure memory usage (separate run to avoid timing contamination)
    peak_memory_kb, memory_result = measure_memory(algorithm, data)
    
    # Verify correctness using time measurement result
    is_correct = verify_correctness(time_result, data)
    
    return BenchmarkResult(
        algorithm_name=algorithm_name,
        distribution=distribution_name,
        input_size=input_size,
        execution_time=execution_time,
        peak_memory_kb=peak_memory_kb,
        is_correct=is_correct
    )


def run_benchmark_suite(
    algorithms: Dict[str, Callable],
    data_generators: Dict[str, Callable],
    input_sizes: List[int],
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Execute comprehensive benchmarking across all algorithms and distributions.
    
    Performs a full factorial experiment design, testing every combination of:
        - Algorithm (Bubble, Merge, Quick, Timsort)
        - Distribution (Random, Sorted, Reverse)
        - Input size (configurable range)
    
    This systematic approach enables:
        - Cross-algorithm performance comparison
        - Distribution sensitivity analysis
        - Scalability characterization (time/space complexity validation)
    
    Args:
        algorithms: Dictionary mapping algorithm names to functions
        data_generators: Dictionary mapping distribution names to generators
        input_sizes: List of array sizes to test (e.g., [100, 500, 1000])
        verbose: Print progress updates if True
        
    Returns:
        List of BenchmarkResult objects for all experiments
        
    Example:
        >>> from algorithms import ALGORITHMS
        >>> from data_utils import DATA_GENERATORS
        >>> results = run_benchmark_suite(
        ...     ALGORITHMS,
        ...     DATA_GENERATORS,
        ...     input_sizes=[100, 500, 1000],
        ...     verbose=True
        ... )
        Running benchmarks...
        [Bubble Sort, Random, n=100] ✓
        ...
        >>> len(results)
        36  # 4 algorithms × 3 distributions × 3 sizes
        
    Note:
        Progress is printed to stdout when verbose=True.
        Large input sizes may cause exponential runtime for O(n²) algorithms.
    """
    results = []
    total_experiments = len(algorithms) * len(data_generators) * len(input_sizes)
    current_experiment = 0
    
    if verbose:
        print(f"Running {total_experiments} benchmark experiments...")
        print("=" * 70)
    
    for algo_name, algo_func in algorithms.items():
        for dist_name, data_gen in data_generators.items():
            for size in input_sizes:
                current_experiment += 1
                
                # Generate data for this experiment
                data = data_gen(size)
                
                # Run benchmark
                result = benchmark_algorithm(
                    algo_func,
                    algo_name,
                    data,
                    dist_name
                )
                
                results.append(result)
                
                # Progress reporting
                if verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(
                        f"[{current_experiment}/{total_experiments}] "
                        f"{algo_name:12} | {dist_name:12} | n={size:5} | "
                        f"{result.execution_time:8.6f}s | "
                        f"{result.peak_memory_kb:8.2f} KB {status}"
                    )
    
    if verbose:
        print("=" * 70)
        print("Benchmark suite completed successfully!")
    
    return results


if __name__ == "__main__":
    # Minimal validation test
    from algorithms import ALGORITHMS
    from data_utils import DATA_GENERATORS
    
    print("Benchmark Module Validation")
    print("=" * 70)
    
    # Test with small dataset
    test_sizes = [10, 50, 100]
    
    results = run_benchmark_suite(
        algorithms=ALGORITHMS,
        data_generators=DATA_GENERATORS,
        input_sizes=test_sizes,
        verbose=True
    )
    
    # Summary statistics
    print("\nValidation Summary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  All correct: {all(r.is_correct for r in results)}")
    print(f"  Fastest: {min(results, key=lambda r: r.execution_time).algorithm_name}")
    print(f"  Most memory: {max(results, key=lambda r: r.peak_memory_kb).algorithm_name}")

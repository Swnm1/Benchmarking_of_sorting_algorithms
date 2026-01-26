
"""
Sorting Algorithm Implementations for Empirical Complexity Analysis

This module provides academic implementations of classical sorting algorithms
alongside a production-grade baseline (Timsort) for comparative benchmarking.

Complexity Analysis:
    - Bubble Sort: O(n²) time, O(1) auxiliary space
    - Merge Sort: O(n log n) time, O(n) auxiliary space
    - Quick Sort: O(n log n) average, O(n²) worst-case time, O(log n) space
    - Timsort: O(n log n) time, hybrid adaptive algorithm (Python default)

Author: Swornim Shrestha
Date: January 2026
References:
    - Cormen, T. H., et al. (2009). Introduction to Algorithms (3rd ed.)
    - Angela1c CTA Benchmarking: https://www.angela1c.com/projects/cta_benchmarking/
    - SortBenchmarker: https://github.com/sanaynesargi/SortBenchmarker
"""

from typing import List, TypeVar

T = TypeVar('T')


def bubble_sort(arr: List[T]) -> List[T]:
    """
    Bubble Sort: Educational baseline demonstrating O(n²) complexity.
    
    This implementation serves as a pedagogical control, illustrating
    quadratic growth behavior. Successive passes "bubble" larger elements
    toward the end of the array through adjacent comparisons.
    
    Time Complexity:
        Best: O(n) with early termination optimization
        Average: O(n²)
        Worst: O(n²)
    
    Space Complexity: O(1) auxiliary space (in-place)
    
    Args:
        arr: Input list to be sorted
        
    Returns:
        Sorted list in ascending order
        
    Example:
        >>> bubble_sort([64, 34, 25, 12, 22, 11, 90])
        [11, 12, 22, 25, 34, 64, 90]
    """
    n = len(arr)
    result = arr.copy()  # Maintain immutability for benchmarking
    
    for i in range(n):
        swapped = False
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        
        # Early termination if no swaps occurred
        if not swapped:
            break
    
    return result


def merge_sort(arr: List[T]) -> List[T]:
    """
    Merge Sort: Divide-and-conquer algorithm with O(n log n) complexity.
    
    Recursively divides the array into halves, sorts each half, and merges
    the sorted subarrays. This implementation demonstrates O(n) auxiliary
    space requirements due to the merge operation.
    
    Time Complexity:
        Best: O(n log n)
        Average: O(n log n)
        Worst: O(n log n)
    
    Space Complexity: O(n) auxiliary space for merge arrays
    
    Args:
        arr: Input list to be sorted
        
    Returns:
        Sorted list in ascending order
        
    Example:
        >>> merge_sort([38, 27, 43, 3, 9, 82, 10])
        [3, 9, 10, 27, 38, 43, 82]
    """
    if len(arr) <= 1:
        return arr.copy()
    
    # Divide phase
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer phase: merge sorted halves
    return _merge(left, right)


def _merge(left: List[T], right: List[T]) -> List[T]:
    """
    Merge two sorted arrays into a single sorted array.
    
    Args:
        left: First sorted subarray
        right: Second sorted subarray
        
    Returns:
        Merged sorted array
    """
    result = []
    i = j = 0
    
    # Merge elements in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def quick_sort(arr: List[T]) -> List[T]:
    """
    Quick Sort: Efficient in-place sorting with Median-of-Three pivot selection.
    
    This implementation employs the Median-of-Three pivot strategy to mitigate
    worst-case O(n²) behavior on already-sorted or reverse-sorted inputs.
    The algorithm partitions around a pivot and recursively sorts partitions.
    
    Time Complexity:
        Best: O(n log n)
        Average: O(n log n)
        Worst: O(n²) - mitigated by pivot strategy
    
    Space Complexity: O(log n) for recursion stack (average case)
    
    Args:
        arr: Input list to be sorted
        
    Returns:
        Sorted list in ascending order
        
    Example:
        >>> quick_sort([10, 7, 8, 9, 1, 5])
        [1, 5, 7, 8, 9, 10]
    """
    result = arr.copy()
    _quick_sort_helper(result, 0, len(result) - 1)
    return result


def _quick_sort_helper(arr: List[T], low: int, high: int) -> None:
    """
    Recursive helper for Quick Sort with in-place partitioning.
    
    Args:
        arr: Array to sort (modified in-place)
        low: Starting index of partition
        high: Ending index of partition
    """
    if low < high:
        # Partition and get pivot index
        pivot_idx = _partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        _quick_sort_helper(arr, low, pivot_idx - 1)
        _quick_sort_helper(arr, pivot_idx + 1, high)


def _median_of_three(arr: List[T], low: int, high: int) -> int:
    """
    Median-of-Three pivot selection strategy.
    
    Selects the median of the first, middle, and last elements as pivot
    to improve performance on partially sorted data.
    
    Args:
        arr: Input array
        low: Starting index
        high: Ending index
        
    Returns:
        Index of the median element
    """
    mid = (low + high) // 2
    
    # Sort low, mid, high positions to find median
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    
    # Place median at high position for partitioning
    arr[mid], arr[high] = arr[high], arr[mid]
    return high


def _partition(arr: List[T], low: int, high: int) -> int:
    """
    Lomuto partition scheme with Median-of-Three pivot.
    
    Args:
        arr: Array to partition (modified in-place)
        low: Starting index
        high: Ending index
        
    Returns:
        Final position of pivot element
    """
    # Use Median-of-Three for pivot selection
    _median_of_three(arr, low, high)
    pivot = arr[high]
    
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # Place pivot in correct position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def timsort(arr: List[T]) -> List[T]:
    """
    Timsort: Production-grade baseline using Python's built-in sorted().
    
    Timsort is a hybrid adaptive sorting algorithm derived from merge sort
    and insertion sort. It is the default sorting algorithm in Python (since 2.3)
    and Java (since SE 7), optimized for real-world data with partially ordered
    sequences.
    
    This wrapper serves as a production baseline for empirical comparison,
    demonstrating the performance gap between academic implementations and
    industrial-strength algorithms.
    
    Time Complexity:
        Best: O(n) on nearly sorted data
        Average: O(n log n)
        Worst: O(n log n)
    
    Space Complexity: O(n) auxiliary space
    
    Args:
        arr: Input list to be sorted
        
    Returns:
        Sorted list in ascending order
        
    Reference:
        Peters, T. (2002). "Timsort" - Python's sorting algorithm
    
    Example:
        >>> timsort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
    """
    return sorted(arr)


# Algorithm registry for benchmarking framework
ALGORITHMS = {
    'Bubble Sort': bubble_sort,
    'Merge Sort': merge_sort,
    'Quick Sort': quick_sort,
    'Timsort': timsort,
}


if __name__ == "__main__":
    # Validation suite
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 2, 8, 1, 9],
        [1, 2, 3, 4, 5],  # Already sorted
        [5, 4, 3, 2, 1],  # Reverse sorted
        [42],  # Single element
        [],  # Empty array
    ]
    
    print("Algorithm Validation Suite")
    print("=" * 50)
    
    for name, algo in ALGORITHMS.items():
        print(f"\n{name}:")
        for test in test_cases:
            result = algo(test)
            expected = sorted(test)
            status = "✓" if result == expected else "✗"
            print(f"  {status} Input: {test[:5]}{'...' if len(test) > 5 else ''} → {result[:5]}{'...' if len(result) > 5 else ''}")

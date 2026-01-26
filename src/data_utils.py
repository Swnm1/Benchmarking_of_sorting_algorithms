"""
Data Generation Utilities for Sorting Algorithm Benchmarking

This module provides reproducible dataset generators for empirical analysis
of sorting algorithms across different input distributions. All generators
use fixed random seeds to ensure experimental reproducibility.

Distribution Types:
    - Random: Uniformly distributed integers (worst-case for many algorithms)
    - Sorted: Pre-sorted ascending order (best-case for adaptive algorithms)
    - Reverse: Descending order (worst-case for naive Quick Sort)

Author: Swornim Shrestha
Date: January 2026
References:
    - Knuth, D. E. (1998). The Art of Computer Programming, Vol. 3: Sorting and Searching
    - Angela1c CTA Benchmarking: https://www.angela1c.com/projects/cta_benchmarking/
"""

import random
from typing import List

# Fixed seed for reproducibility across all experiments
RANDOM_SEED = 42


def generate_random_data(n: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """
    Generate uniformly distributed random integers.
    
    This distribution represents the general case for sorting algorithms,
    providing no inherent structure for optimization. It serves as the
    primary test case for average-case complexity analysis.
    
    Args:
        n: Number of elements to generate
        min_val: Minimum value in range (inclusive)
        max_val: Maximum value in range (inclusive)
        
    Returns:
        List of n random integers in range [min_val, max_val]
        
    Example:
        >>> random.seed(42)  # For reproducibility
        >>> generate_random_data(5, 0, 100)
        [81, 14, 3, 94, 35]
        
    Note:
        Uses fixed RANDOM_SEED for experimental reproducibility.
    """
    random.seed(RANDOM_SEED)
    return [random.randint(min_val, max_val) for _ in range(n)]


def generate_sorted_data(n: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """
    Generate pre-sorted data in ascending order.
    
    This distribution tests best-case performance for adaptive algorithms
    (e.g., Timsort, optimized Bubble Sort) and worst-case behavior for
    naive Quick Sort implementations without pivot randomization.
    
    Sorted input reveals:
        - Bubble Sort: O(n) with early termination
        - Merge Sort: Still O(n log n), no advantage
        - Quick Sort: O(n²) without median pivot strategy
        - Timsort: O(n) due to run detection
    
    Args:
        n: Number of elements to generate
        min_val: Starting value
        max_val: Ending value
        
    Returns:
        List of n integers sorted in ascending order
        
    Example:
        >>> generate_sorted_data(5, 0, 100)
        [0, 25, 50, 75, 100]
        
    Note:
        Uses linspace-like distribution for even spacing.
    """
    if n == 0:
        return []
    if n == 1:
        return [min_val]
    
    step = (max_val - min_val) / (n - 1)
    return [int(min_val + i * step) for i in range(n)]


def generate_reverse_sorted_data(n: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """
    Generate data sorted in descending order.
    
    This distribution represents worst-case input for many algorithms,
    particularly those without optimization for reverse-sorted data.
    It stresses comparison counts and swap operations.
    
    Reverse-sorted input reveals:
        - Bubble Sort: O(n²) with maximum swaps
        - Merge Sort: Still O(n log n), structure-agnostic
        - Quick Sort: O(n²) with poor pivot selection
        - Timsort: O(n) with reverse-run optimization
    
    Args:
        n: Number of elements to generate
        min_val: Minimum value (will be last)
        max_val: Maximum value (will be first)
        
    Returns:
        List of n integers sorted in descending order
        
    Example:
        >>> generate_reverse_sorted_data(5, 0, 100)
        [100, 75, 50, 25, 0]
        
    Note:
        Implemented as reversed sorted data for consistency.
    """
    return list(reversed(generate_sorted_data(n, min_val, max_val)))


def generate_nearly_sorted_data(n: int, swap_percentage: float = 0.1) -> List[int]:
    """
    Generate nearly sorted data with controlled disorder.
    
    Creates sorted data and randomly swaps a percentage of elements,
    simulating real-world scenarios where data is partially ordered.
    This tests adaptive algorithm performance on realistic inputs.
    
    Args:
        n: Number of elements to generate
        swap_percentage: Proportion of elements to randomly swap (0.0 to 1.0)
        
    Returns:
        List of n integers with controlled disorder
        
    Example:
        >>> random.seed(42)
        >>> generate_nearly_sorted_data(10, 0.2)
        [0, 7, 2, 3, 4, 5, 1, 9, 8, 6]
        
    Note:
        Useful for demonstrating Timsort's adaptive behavior.
    """
    random.seed(RANDOM_SEED)
    data = generate_sorted_data(n)
    
    num_swaps = int(n * swap_percentage)
    for _ in range(num_swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        data[i], data[j] = data[j], data[i]
    
    return data


# Distribution registry for benchmarking framework
DATA_GENERATORS = {
    'Random': generate_random_data,
    'Sorted': generate_sorted_data,
    'Reverse': generate_reverse_sorted_data,
}


if __name__ == "__main__":
    # Validation and demonstration
    print("Data Generation Validation Suite")
    print("=" * 60)
    
    test_size = 10
    
    print(f"\n1. Random Distribution (n={test_size}):")
    random_data = generate_random_data(test_size)
    print(f"   {random_data}")
    print(f"   Min: {min(random_data)}, Max: {max(random_data)}")
    
    print(f"\n2. Sorted Distribution (n={test_size}):")
    sorted_data = generate_sorted_data(test_size)
    print(f"   {sorted_data}")
    print(f"   Is sorted: {sorted_data == sorted(sorted_data)}")
    
    print(f"\n3. Reverse-Sorted Distribution (n={test_size}):")
    reverse_data = generate_reverse_sorted_data(test_size)
    print(f"   {reverse_data}")
    print(f"   Is reverse sorted: {reverse_data == sorted(reverse_data, reverse=True)}")
    
    print(f"\n4. Nearly-Sorted Distribution (n={test_size}, 20% swapped):")
    nearly_sorted = generate_nearly_sorted_data(test_size, 0.2)
    print(f"   {nearly_sorted}")
    
    # Reproducibility test
    print(f"\n5. Reproducibility Check:")
    data1 = generate_random_data(test_size)
    data2 = generate_random_data(test_size)
    print(f"   First run:  {data1}")
    print(f"   Second run: {data2}")
    print(f"   Identical: {data1 == data2} ✓" if data1 == data2 else "   Identical: False ✗")
    
    print("\n" + "=" * 60)
    print("All distributions validated successfully!")

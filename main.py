"""
Empirical Evaluation of Sorting Algorithms: Main Execution Script

This script orchestrates the complete benchmarking pipeline for sorting
algorithm analysis, including data generation, performance measurement,
statistical analysis, and publication-quality visualization.

Pipeline Stages:
    1. Experiment Configuration: Define input sizes and parameters
    2. Benchmark Execution: Run full factorial experiment design
    3. Data Export: Save raw results to CSV for reproducibility
    4. Visualization: Generate time and memory complexity plots
    5. Analysis: Print summary statistics and insights

Output Artifacts:
    - results.csv: Raw benchmark data for external analysis
    - plots/execution_time.png: Time complexity growth curves
    - plots/memory_usage.png: Space complexity visualization

Author: Swornim Shestha
Date: January 2026
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from src.algorithms import ALGORITHMS
from src.data_utils import DATA_GENERATORS
from src.benchmark import run_benchmark_suite, BenchmarkResult


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Input sizes for scalability analysis
# Adjust based on available compute time (O(n²) algorithms may be slow)
INPUT_SIZES = [100, 500, 1000, 2000, 3000, 5000]

# Output directory for plots
PLOTS_DIR = "plots"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_plots_directory():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        print(f"Created directory: {PLOTS_DIR}/")


def export_to_csv(results: List[BenchmarkResult], filename: str = "results.csv"):
    """
    Export benchmark results to CSV for external analysis and reproducibility.
    
    Args:
        results: List of BenchmarkResult objects
        filename: Output CSV filename
        
    Note:
        CSV format enables further analysis in R, Excel, or Jupyter notebooks.
    """
    # Convert results to structured records
    records = [
        {
            'Algorithm': r.algorithm_name,
            'Distribution': r.distribution,
            'Input_Size': r.input_size,
            'Execution_Time_s': r.execution_time,
            'Peak_Memory_KB': r.peak_memory_kb,
            'Correct': r.is_correct
        }
        for r in results
    ]
    
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"\n✓ Results exported to: {filename}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")


def print_summary_statistics(results: List[BenchmarkResult]):
    """
    Print descriptive statistics and key insights from benchmark results.
    
    Args:
        results: List of BenchmarkResult objects
    """
    df = pd.DataFrame([
        {
            'Algorithm': r.algorithm_name,
            'Distribution': r.distribution,
            'Input_Size': r.input_size,
            'Execution_Time_s': r.execution_time,
            'Peak_Memory_KB': r.peak_memory_kb,
        }
        for r in results
    ])
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Overall fastest algorithm
    fastest = df.loc[df['Execution_Time_s'].idxmin()]
    print(f"\n⚡ Fastest Overall:")
    print(f"   {fastest['Algorithm']} on {fastest['Distribution']} (n={fastest['Input_Size']})")
    print(f"   Time: {fastest['Execution_Time_s']:.6f}s")
    
    # Most memory-efficient
    least_memory = df.loc[df['Peak_Memory_KB'].idxmin()]
    print(f"\n💾 Most Memory Efficient:")
    print(f"   {least_memory['Algorithm']} on {least_memory['Distribution']} (n={least_memory['Input_Size']})")
    print(f"   Memory: {least_memory['Peak_Memory_KB']:.2f} KB")
    
    # Average time by algorithm
    print(f"\n📊 Average Execution Time by Algorithm:")
    avg_time = df.groupby('Algorithm')['Execution_Time_s'].mean().sort_values()
    for algo, time in avg_time.items():
        print(f"   {algo:12}: {time:.6f}s")
    
    # Average memory by algorithm
    print(f"\n📊 Average Memory Usage by Algorithm:")
    avg_memory = df.groupby('Algorithm')['Peak_Memory_KB'].mean().sort_values()
    for algo, memory in avg_memory.items():
        print(f"   {algo:12}: {memory:.2f} KB")
    
    print("\n" + "=" * 70)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_execution_time(results: List[BenchmarkResult]):
    """
    Generate execution time complexity plots.
    
    Creates subplots for each data distribution showing time growth
    as a function of input size. This visualization reveals:
        - O(n²) quadratic growth (Bubble Sort)
        - O(n log n) linearithmic growth (Merge, Quick, Timsort)
        - Distribution sensitivity (sorted vs. random vs. reverse)
    
    Args:
        results: List of BenchmarkResult objects
        
    Output:
        Saves plot to plots/execution_time.png
    """
    # Convert to DataFrame for easier grouping
    df = pd.DataFrame([
        {
            'Algorithm': r.algorithm_name,
            'Distribution': r.distribution,
            'Input_Size': r.input_size,
            'Execution_Time_s': r.execution_time,
        }
        for r in results
    ])
    
    # Create subplots for each distribution
    distributions = df['Distribution'].unique()
    fig, axes = plt.subplots(1, len(distributions), figsize=(18, 5))
    
    if len(distributions) == 1:
        axes = [axes]
    
    # Color scheme for algorithms
    colors = {
        'Bubble Sort': '#e74c3c',  # Red - slowest
        'Merge Sort': '#3498db',   # Blue
        'Quick Sort': '#2ecc71',   # Green
        'Timsort': '#9b59b6'       # Purple - fastest
    }
    
    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        dist_data = df[df['Distribution'] == dist]
        
        # Plot each algorithm
        for algo in df['Algorithm'].unique():
            algo_data = dist_data[dist_data['Algorithm'] == algo].sort_values('Input_Size')
            ax.plot(
                algo_data['Input_Size'],
                algo_data['Execution_Time_s'],
                marker='o',
                label=algo,
                color=colors.get(algo, '#95a5a6'),
                linewidth=2,
                markersize=6
            )
        
        ax.set_title(f'{dist} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Size (n)', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')  # Log scale reveals complexity classes
    
    plt.suptitle(
        'Execution Time Growth: Empirical Complexity Analysis',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, 'execution_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Execution time plot saved to: {output_path}")
    plt.close()


def plot_memory_usage(results: List[BenchmarkResult]):
    """
    Generate memory usage complexity plots.
    
    Visualizes peak memory consumption across algorithms and input sizes.
    This reveals space complexity differences:
        - O(1) in-place algorithms (Bubble Sort)
        - O(n) auxiliary space (Merge Sort)
        - O(log n) recursion stack (Quick Sort)
    
    Args:
        results: List of BenchmarkResult objects
        
    Output:
        Saves plot to plots/memory_usage.png
    """
    df = pd.DataFrame([
        {
            'Algorithm': r.algorithm_name,
            'Distribution': r.distribution,
            'Input_Size': r.input_size,
            'Peak_Memory_KB': r.peak_memory_kb,
        }
        for r in results
    ])
    
    distributions = df['Distribution'].unique()
    fig, axes = plt.subplots(1, len(distributions), figsize=(18, 5))
    
    if len(distributions) == 1:
        axes = [axes]
    
    colors = {
        'Bubble Sort': '#e74c3c',
        'Merge Sort': '#3498db',
        'Quick Sort': '#2ecc71',
        'Timsort': '#9b59b6'
    }
    
    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        dist_data = df[df['Distribution'] == dist]
        
        for algo in df['Algorithm'].unique():
            algo_data = dist_data[dist_data['Algorithm'] == algo].sort_values('Input_Size')
            ax.plot(
                algo_data['Input_Size'],
                algo_data['Peak_Memory_KB'],
                marker='s',
                label=algo,
                color=colors.get(algo, '#95a5a6'),
                linewidth=2,
                markersize=6
            )
        
        ax.set_title(f'{dist} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Size (n)', fontsize=12)
        ax.set_ylabel('Peak Memory Usage (KB)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(
        'Memory Usage Growth: Space Complexity Analysis',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, 'memory_usage.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Memory usage plot saved to: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline for sorting algorithm benchmarking.
    
    Orchestrates:
        1. Directory setup
        2. Benchmark execution
        3. Data export
        4. Visualization generation
        5. Statistical summary
    """
    print("\n" + "=" * 70)
    print("EMPIRICAL EVALUATION OF SORTING ALGORITHMS")
    print("Complexity, Memory, and Scalability Analysis")
    print("=" * 70)
    
    # Setup
    ensure_plots_directory()
    
    # Execute benchmark suite
    print(f"\nConfiguration:")
    print(f"  Algorithms: {len(ALGORITHMS)}")
    print(f"  Distributions: {len(DATA_GENERATORS)}")
    print(f"  Input sizes: {INPUT_SIZES}")
    print(f"  Total experiments: {len(ALGORITHMS) * len(DATA_GENERATORS) * len(INPUT_SIZES)}")
    print()
    
    results = run_benchmark_suite(
        algorithms=ALGORITHMS,
        data_generators=DATA_GENERATORS,
        input_sizes=INPUT_SIZES,
        verbose=True
    )
    
    # Export and analyze
    export_to_csv(results)
    print_summary_statistics(results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_execution_time(results)
    plot_memory_usage(results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ Benchmarking pipeline completed successfully!")
    print("\nGenerated artifacts:")
    print("  - results.csv (raw data)")
    print("  - plots/execution_time.png")
    print("  - plots/memory_usage.png")
    print("\nNext steps:")
    print("  1. Review plots/ directory for visualizations")
    print("  2. Analyze results.csv for detailed metrics")
    print("  3. Update README.md with findings and insights")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

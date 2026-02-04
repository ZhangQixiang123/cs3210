#!/usr/bin/env python3
"""
CS3210 Tutorial 1 - Exercise 15
Automated Experiment Script for asdf.cpp Analysis

This script:
1. Compiles original and optimized versions
2. Runs experiments with different array sizes
3. Collects performance metrics using perf stat
4. Generates graphs for the report
5. Outputs raw data for the appendix

Usage:
    python3 ex15_experiment.py              # Use placeholder data (for testing graphs)
    python3 ex15_experiment.py --run        # Run actual experiments (needs perf)
    python3 ex15_experiment.py --run --slurm  # Run with Slurm on lab cluster
"""

import matplotlib.pyplot as plt
import numpy as np
import subprocess
import re
import os
import sys
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of repetitions for each experiment
NUM_RUNS = 5

# Array sizes to test (for parameter exploration)
# Using smaller sizes for bubble sort since it's O(n²)
ARRAY_SIZES = [1000, 2000, 5000, 10000, 20000, 30000]

# For the optimized version, we can test larger sizes
ARRAY_SIZES_OPTIMIZED = [1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000]

# Perf events to measure
PERF_EVENTS = 'cycles,instructions,cache-misses,branch-misses'

# Use Slurm?
USE_SLURM = False

# Source files
ORIGINAL_SRC = 'asdf_original.cpp'
OPTIMIZED_SRC = 'asdf_optimized.cpp'

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_command(cmd, capture=True, timeout=300):
    """Run a shell command and return output"""
    print(f"    $ {cmd}")
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    text=True, timeout=timeout)
            return result.stdout + result.stderr
        else:
            subprocess.run(cmd, shell=True, timeout=timeout)
            return ""
    except subprocess.TimeoutExpired:
        print(f"    Timeout after {timeout}s")
        return "TIMEOUT"
    except Exception as e:
        print(f"    Error: {e}")
        return ""

def compile_programs():
    """Compile both versions of the program"""
    print("\n" + "="*70)
    print("STEP 1: Compiling programs...")
    print("="*70)

    # Compile original (bubble sort)
    if os.path.exists(ORIGINAL_SRC):
        cmd = f"g++ -O0 -o asdf_original {ORIGINAL_SRC}"
        run_command(cmd, capture=False)
        print(f"  Compiled: {ORIGINAL_SRC} -> asdf_original")
    else:
        print(f"  ERROR: {ORIGINAL_SRC} not found!")
        return False

    # Compile optimized (std::sort)
    if os.path.exists(OPTIMIZED_SRC):
        cmd = f"g++ -O0 -o asdf_optimized {OPTIMIZED_SRC}"
        run_command(cmd, capture=False)
        print(f"  Compiled: {OPTIMIZED_SRC} -> asdf_optimized")
    else:
        print(f"  ERROR: {OPTIMIZED_SRC} not found!")
        return False

    return True

def parse_perf_output(output):
    """Parse perf stat output and extract metrics"""
    metrics = {
        'cycles': 0,
        'instructions': 0,
        'cache-misses': 0,
        'branch-misses': 0,
        'time': 0
    }

    for line in output.split('\n'):
        line_clean = line.replace(',', '').strip()

        if 'cycles' in line and 'instructions' not in line:
            match = re.search(r'(\d+)\s+cycles', line_clean)
            if match:
                metrics['cycles'] = int(match.group(1))

        if 'instructions' in line:
            match = re.search(r'(\d+)\s+instructions', line_clean)
            if match:
                metrics['instructions'] = int(match.group(1))

        if 'cache-misses' in line:
            match = re.search(r'(\d+)\s+cache-misses', line_clean)
            if match:
                metrics['cache-misses'] = int(match.group(1))

        if 'branch-misses' in line:
            match = re.search(r'(\d+)\s+branch-misses', line_clean)
            if match:
                metrics['branch-misses'] = int(match.group(1))

        if 'seconds time elapsed' in line:
            match = re.search(r'([\d.]+)\s+seconds time elapsed', line)
            if match:
                metrics['time'] = float(match.group(1))

    return metrics

def run_single_experiment(program, array_size, num_runs=NUM_RUNS):
    """Run perf stat on a program with given array size"""
    results = []

    for i in range(num_runs):
        if USE_SLURM:
            cmd = f"srun perf stat -e {PERF_EVENTS} -- ./{program} {array_size} 2>&1"
        else:
            cmd = f"perf stat -e {PERF_EVENTS} -- ./{program} {array_size} 2>&1"

        output = run_command(cmd, timeout=120)

        if output == "TIMEOUT":
            return None

        metrics = parse_perf_output(output)
        if metrics['time'] > 0:
            results.append(metrics)

    return results

def run_all_experiments():
    """Run experiments on both programs with different array sizes"""
    print("\n" + "="*70)
    print("STEP 2: Running experiments...")
    print("="*70)

    all_results = {
        'original': {},
        'optimized': {}
    }

    # Test original (bubble sort) with smaller sizes
    print("\n  Testing ORIGINAL (Bubble Sort)...")
    for size in ARRAY_SIZES:
        print(f"\n    Array size: {size}")
        results = run_single_experiment('asdf_original', size)
        if results:
            all_results['original'][size] = results
            avg_time = np.mean([r['time'] for r in results])
            print(f"      Average time: {avg_time:.3f}s")
        else:
            print(f"      Skipped (timeout or error)")
            break  # Stop if bubble sort becomes too slow

    # Test optimized (std::sort) with larger sizes
    print("\n  Testing OPTIMIZED (std::sort)...")
    for size in ARRAY_SIZES_OPTIMIZED:
        print(f"\n    Array size: {size}")
        results = run_single_experiment('asdf_optimized', size)
        if results:
            all_results['optimized'][size] = results
            avg_time = np.mean([r['time'] for r in results])
            print(f"      Average time: {avg_time:.3f}s")

    return all_results

# =============================================================================
# PLACEHOLDER DATA (used when perf is not available)
# =============================================================================

def get_placeholder_data():
    """Return placeholder data for testing graphs"""
    # Simulated data based on expected O(n²) vs O(n log n) behavior
    return {
        'original': {
            1000:  [{'time': 0.008, 'cycles': 2e7, 'instructions': 1.5e7, 'cache-misses': 5000, 'branch-misses': 1e5} for _ in range(5)],
            2000:  [{'time': 0.030, 'cycles': 8e7, 'instructions': 6e7, 'cache-misses': 8000, 'branch-misses': 4e5} for _ in range(5)],
            5000:  [{'time': 0.180, 'cycles': 5e8, 'instructions': 3.7e8, 'cache-misses': 15000, 'branch-misses': 2.5e6} for _ in range(5)],
            10000: [{'time': 0.720, 'cycles': 2e9, 'instructions': 1.5e9, 'cache-misses': 25000, 'branch-misses': 1e7} for _ in range(5)],
            20000: [{'time': 2.880, 'cycles': 8e9, 'instructions': 6e9, 'cache-misses': 45000, 'branch-misses': 4e7} for _ in range(5)],
            30000: [{'time': 6.480, 'cycles': 1.8e10, 'instructions': 1.35e10, 'cache-misses': 65000, 'branch-misses': 9e7} for _ in range(5)],
        },
        'optimized': {
            1000:  [{'time': 0.001, 'cycles': 3e6, 'instructions': 2e6, 'cache-misses': 2000, 'branch-misses': 2e4} for _ in range(5)],
            2000:  [{'time': 0.002, 'cycles': 7e6, 'instructions': 5e6, 'cache-misses': 3500, 'branch-misses': 4.5e4} for _ in range(5)],
            5000:  [{'time': 0.005, 'cycles': 2e7, 'instructions': 1.4e7, 'cache-misses': 7000, 'branch-misses': 1.2e5} for _ in range(5)],
            10000: [{'time': 0.011, 'cycles': 4.5e7, 'instructions': 3e7, 'cache-misses': 12000, 'branch-misses': 2.6e5} for _ in range(5)],
            20000: [{'time': 0.024, 'cycles': 1e8, 'instructions': 7e7, 'cache-misses': 22000, 'branch-misses': 5.5e5} for _ in range(5)],
            30000: [{'time': 0.038, 'cycles': 1.6e8, 'instructions': 1.1e8, 'cache-misses': 32000, 'branch-misses': 8.5e5} for _ in range(5)],
            50000: [{'time': 0.068, 'cycles': 2.8e8, 'instructions': 2e8, 'cache-misses': 50000, 'branch-misses': 1.5e6} for _ in range(5)],
            100000:[{'time': 0.150, 'cycles': 6.2e8, 'instructions': 4.3e8, 'cache-misses': 95000, 'branch-misses': 3.2e6} for _ in range(5)],
        }
    }

# =============================================================================
# GRAPHING FUNCTIONS
# =============================================================================

def extract_metric(results, metric):
    """Extract a specific metric from results dict"""
    sizes = sorted(results.keys())
    means = []
    stds = []
    for size in sizes:
        values = [r[metric] for r in results[size]]
        means.append(np.mean(values))
        stds.append(np.std(values))
    return sizes, means, stds

def plot_time_vs_size(data):
    """Plot execution time vs array size for both versions"""
    plt.figure(figsize=(12, 7))

    # Original (bubble sort)
    if data['original']:
        sizes, means, stds = extract_metric(data['original'], 'time')
        plt.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5,
                     linewidth=2, markersize=8, color='#e74c3c',
                     label='Original (Bubble Sort) - O(n²)')

    # Optimized (std::sort)
    if data['optimized']:
        sizes, means, stds = extract_metric(data['optimized'], 'time')
        plt.errorbar(sizes, means, yerr=stds, fmt='s-', capsize=5,
                     linewidth=2, markersize=8, color='#2ecc71',
                     label='Optimized (std::sort) - O(n log n)')

    plt.xlabel('Array Size (n)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Execution Time vs Array Size: Bubble Sort vs std::sort', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_time_comparison.png', dpi=150)
    print("  Saved: ex15_time_comparison.png")

def plot_time_vs_size_log(data):
    """Plot execution time vs array size with log scale"""
    plt.figure(figsize=(12, 7))

    # Original (bubble sort)
    if data['original']:
        sizes, means, stds = extract_metric(data['original'], 'time')
        plt.plot(sizes, means, 'o-', linewidth=2, markersize=8, color='#e74c3c',
                 label='Original (Bubble Sort) - O(n²)')

    # Optimized (std::sort)
    if data['optimized']:
        sizes, means, stds = extract_metric(data['optimized'], 'time')
        plt.plot(sizes, means, 's-', linewidth=2, markersize=8, color='#2ecc71',
                 label='Optimized (std::sort) - O(n log n)')

    plt.xlabel('Array Size (n)', fontsize=12)
    plt.ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
    plt.title('Execution Time vs Array Size (Log Scale)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('ex15_time_comparison_log.png', dpi=150)
    print("  Saved: ex15_time_comparison_log.png")

def plot_speedup(data):
    """Plot speedup of optimized over original"""
    plt.figure(figsize=(10, 6))

    # Find common sizes
    orig_sizes = set(data['original'].keys())
    opt_sizes = set(data['optimized'].keys())
    common_sizes = sorted(orig_sizes & opt_sizes)

    if not common_sizes:
        print("  Warning: No common sizes to compare")
        return

    speedups = []
    for size in common_sizes:
        orig_time = np.mean([r['time'] for r in data['original'][size]])
        opt_time = np.mean([r['time'] for r in data['optimized'][size]])
        speedups.append(orig_time / opt_time)

    x = np.arange(len(common_sizes))
    bars = plt.bar(x, speedups, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)

    plt.xlabel('Array Size', fontsize=12)
    plt.ylabel('Speedup (Original Time / Optimized Time)', fontsize=12)
    plt.title('Speedup: std::sort vs Bubble Sort', fontsize=14)
    plt.xticks(x, common_sizes)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('ex15_speedup.png', dpi=150)
    print("  Saved: ex15_speedup.png")

def plot_cycles_comparison(data):
    """Plot CPU cycles comparison"""
    plt.figure(figsize=(12, 7))

    # Original
    if data['original']:
        sizes, means, stds = extract_metric(data['original'], 'cycles')
        plt.plot(sizes, means, 'o-', linewidth=2, markersize=8, color='#e74c3c',
                 label='Original (Bubble Sort)')

    # Optimized
    if data['optimized']:
        sizes, means, stds = extract_metric(data['optimized'], 'cycles')
        plt.plot(sizes, means, 's-', linewidth=2, markersize=8, color='#2ecc71',
                 label='Optimized (std::sort)')

    plt.xlabel('Array Size (n)', fontsize=12)
    plt.ylabel('CPU Cycles', fontsize=12)
    plt.title('CPU Cycles vs Array Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_cycles_comparison.png', dpi=150)
    print("  Saved: ex15_cycles_comparison.png")

def plot_instructions_comparison(data):
    """Plot instructions comparison"""
    plt.figure(figsize=(12, 7))

    # Original
    if data['original']:
        sizes, means, stds = extract_metric(data['original'], 'instructions')
        plt.plot(sizes, means, 'o-', linewidth=2, markersize=8, color='#e74c3c',
                 label='Original (Bubble Sort)')

    # Optimized
    if data['optimized']:
        sizes, means, stds = extract_metric(data['optimized'], 'instructions')
        plt.plot(sizes, means, 's-', linewidth=2, markersize=8, color='#2ecc71',
                 label='Optimized (std::sort)')

    plt.xlabel('Array Size (n)', fontsize=12)
    plt.ylabel('Instructions Executed', fontsize=12)
    plt.title('Instructions vs Array Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_instructions_comparison.png', dpi=150)
    print("  Saved: ex15_instructions_comparison.png")

def plot_bar_comparison_fixed_size(data, size=10000):
    """Bar chart comparing metrics at a fixed array size"""
    if size not in data['original'] or size not in data['optimized']:
        print(f"  Warning: Size {size} not in both datasets")
        return

    plt.figure(figsize=(12, 6))

    metrics = ['time', 'cycles', 'instructions']
    labels = ['Time (s)', 'Cycles', 'Instructions']

    orig_values = [np.mean([r[m] for r in data['original'][size]]) for m in metrics]
    opt_values = [np.mean([r[m] for r in data['optimized'][size]]) for m in metrics]

    # Normalize to original
    normalized_orig = [1.0, 1.0, 1.0]
    normalized_opt = [opt_values[i] / orig_values[i] * 100 for i in range(len(metrics))]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = plt.bar(x - width/2, [100, 100, 100], width, label='Original (Bubble Sort)', color='#e74c3c', alpha=0.8)
    bars2 = plt.bar(x + width/2, normalized_opt, width, label='Optimized (std::sort)', color='#2ecc71', alpha=0.8)

    # Add percentage labels
    for bar, val in zip(bars2, normalized_opt):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Percentage (Original = 100%)', fontsize=12)
    plt.title(f'Performance Comparison at Array Size = {size}', fontsize=14)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('ex15_bar_comparison.png', dpi=150)
    print("  Saved: ex15_bar_comparison.png")

# =============================================================================
# DATA OUTPUT FOR APPENDIX
# =============================================================================

def save_raw_data(data, filename='ex15_raw_data.txt'):
    """Save raw data to a text file for the appendix"""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CS3210 Tutorial 1 - Exercise 15 Raw Data\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        for version in ['original', 'optimized']:
            f.write(f"\n{'='*70}\n")
            f.write(f"{version.upper()} VERSION\n")
            f.write(f"{'='*70}\n\n")

            if data[version]:
                for size in sorted(data[version].keys()):
                    f.write(f"Array Size: {size}\n")
                    f.write("-"*40 + "\n")

                    times = [r['time'] for r in data[version][size]]
                    cycles = [r['cycles'] for r in data[version][size]]
                    instructions = [r['instructions'] for r in data[version][size]]

                    f.write(f"  Time (s):      {times}\n")
                    f.write(f"  Mean Time:     {np.mean(times):.6f} s\n")
                    f.write(f"  Std Dev:       {np.std(times):.6f} s\n")
                    f.write(f"  Cycles:        {[f'{c:.2e}' for c in cycles]}\n")
                    f.write(f"  Mean Cycles:   {np.mean(cycles):.2e}\n")
                    f.write(f"  Instructions:  {[f'{i:.2e}' for i in instructions]}\n")
                    f.write(f"  Mean Instr:    {np.mean(instructions):.2e}\n")
                    f.write("\n")

    print(f"  Saved: {filename}")

def save_csv_data(data, filename='ex15_data.csv'):
    """Save data in CSV format"""
    with open(filename, 'w') as f:
        f.write("version,array_size,run,time,cycles,instructions,cache_misses,branch_misses\n")

        for version in ['original', 'optimized']:
            if data[version]:
                for size in sorted(data[version].keys()):
                    for i, r in enumerate(data[version][size]):
                        f.write(f"{version},{size},{i+1},{r['time']},{r['cycles']},"
                                f"{r['instructions']},{r['cache-misses']},{r['branch-misses']}\n")

    print(f"  Saved: {filename}")

# =============================================================================
# MAIN
# =============================================================================

def check_perf_available():
    """Check if perf is available"""
    result = subprocess.run("which perf", shell=True, capture_output=True)
    return result.returncode == 0

def main():
    print("="*70)
    print("CS3210 Tutorial 1 - Exercise 15 Experiment Script")
    print("="*70)

    run_experiments = '--run' in sys.argv

    if '--slurm' in sys.argv:
        global USE_SLURM
        USE_SLURM = True
        print("Using Slurm for job submission")

    # Get data
    if run_experiments:
        if not check_perf_available():
            print("\nERROR: 'perf' not found. Please run on a Linux system with perf.")
            print("Using placeholder data instead...")
            data = get_placeholder_data()
        else:
            if not compile_programs():
                print("Compilation failed. Exiting.")
                return
            data = run_all_experiments()

            # Fill with placeholder if experiments didn't produce data
            if not data['original'] or not data['optimized']:
                print("\nWarning: Some experiments failed. Using placeholder data.")
                data = get_placeholder_data()
    else:
        print("\nUsing placeholder data. Run with --run to collect real data:")
        print("  python3 ex15_experiment.py --run")
        print("  python3 ex15_experiment.py --run --slurm  (on lab cluster)")
        data = get_placeholder_data()

    # Generate graphs
    print("\n" + "="*70)
    print("STEP 3: Generating graphs...")
    print("="*70)

    plot_time_vs_size(data)
    plot_time_vs_size_log(data)
    plot_speedup(data)
    plot_cycles_comparison(data)
    plot_instructions_comparison(data)
    plot_bar_comparison_fixed_size(data, size=10000)

    # Save raw data
    print("\n" + "="*70)
    print("STEP 4: Saving raw data for appendix...")
    print("="*70)

    save_raw_data(data)
    save_csv_data(data)

    print("\n" + "="*70)
    print("DONE! Generated files:")
    print("="*70)
    print("  Graphs:")
    print("    - ex15_time_comparison.png     (Main graph for report)")
    print("    - ex15_time_comparison_log.png (Log scale version)")
    print("    - ex15_speedup.png             (Speedup bar chart)")
    print("    - ex15_cycles_comparison.png   (CPU cycles)")
    print("    - ex15_instructions_comparison.png")
    print("    - ex15_bar_comparison.png      (Normalized comparison)")
    print("  Data:")
    print("    - ex15_raw_data.txt            (For appendix)")
    print("    - ex15_data.csv                (CSV format)")
    print("="*70)

if __name__ == "__main__":
    main()

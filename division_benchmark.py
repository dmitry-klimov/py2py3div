# -*- coding: utf-8 -*-

"""
Division Benchmark
-----------------
Compares different division implementations across numerical types.
Compatible with both Python 2.x and Python 3.x
"""

from __future__ import print_function, absolute_import
import sys
import time
import random
import platform
import math

try:
    import matplotlib.pyplot as plt
    noPics = False
except ImportError:
    noPics = True

import numpy as np

# Check Python version
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# Try to import optional extensions - check for both Python 2 and 3 compatibility
try:
    import py2py3div_c as c_division

    has_c_extension = True
except ImportError:
    has_c_extension = False

try:
    # Similarly for Cython extension
    import py2py3div_cython as cy_division

    has_cython_extension = True
except ImportError:
    has_cython_extension = False

from py2py3div_python import div_wrapper as python_division

def builtin_division(a, b):
    """Wrapper for built-in division operator"""
    return a / b


def floor_division(a, b):
    """Wrapper for floor division operator"""
    return a // b


def run_benchmark(func, size, data_type, num_runs=5):
    """Run a benchmark and return timing statistics"""

    # Create test data based on data type
    if data_type == 'int':
        numerators = [random.randint(1, 10000) for _ in range(size)]
        # Ensure no zeros for denominators
        denominators = [random.randint(1, 1000) for _ in range(size)]
    elif data_type == 'float':
        numerators = [random.random() * 10000 for _ in range(size)]
        denominators = [random.random() * 1000 + 0.001 for _ in range(size)]  # Avoid values close to zero
    elif data_type == 'mixed':
        numerators = [
            random.choice([random.randint(1, 10000), random.random() * 10000, random.randint(1, 10000) * (10 ** 18)])
            for _ in range(size)]
        denominators = [random.choice(
            [random.randint(1, 1000), random.random() * 1000 + 0.001, random.randint(1, 100) * (10 ** 18)])
                        for _ in range(size)]
    elif data_type == 'huge':
        numerators = [random.randint(1, 10000) * (10 ** 18) for _ in range(size)]
        # Ensure no zeros for denominators
        denominators = [random.randint(1, 1000) * (10 ** 18) for _ in range(size)]
    else:
        raise ValueError("Unsupported data type: {}".format(data_type))

    # Run the benchmark multiple times
    times = []
    results = None  # Store results to verify correctness

    for _ in range(num_runs):
        start_time = time.time()

        results = [func(n, d) for n, d in zip(numerators, denominators)]

        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate statistics
    mean_time = sum(times) / len(times)
    if len(times) > 1:
        # For Python 2 compatibility, manually calculate standard deviation
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        stdev = math.sqrt(variance)
    else:
        stdev = 0

    return {'mean': mean_time, 'stdev': stdev, 'results': results}


def print_results_table(size, results_by_type, data_types):
    """Print benchmark results in a well-formatted table with color-coding"""
    try:
        # For Python 2 and 3 compatibility with colorama
        from colorama import init, Fore, Back, Style
        init(autoreset=True)  # Initialize colorama
        has_colors = True
    except ImportError:
        # Fallback if colorama is not installed
        print("Note: Install 'colorama' for colored output (pip install colorama)")
        has_colors = False

        # Create dummy color constants
        class DummyColors:
            def __getattr__(self, name):
                return ""

        Fore = DummyColors()
        Style = DummyColors()

    # Column headers for the main table - removed floor from competition
    headers = ["Data Type", "Built-in"]

    # Only add Python-impl column if any result has it
    any_has_python_impl = any('python_impl' in result for result in results_by_type)
    if any_has_python_impl:
        headers.extend(["Python-impl", "Py/Built-in"])

    if 'has_c_extension' in globals() and has_c_extension:
        headers.extend(["C-Extension", "C/Built-in"])
    if 'has_cython_extension' in globals() and has_cython_extension:
        headers.extend(["Cython-Ext", "Cy/Built-in"])

    # Add Winner column at the end (excluding built-in and floor)
    headers.append("Winner")

    # Calculate column widths based on header lengths and expected data widths
    col_widths = [max(12, len(h) + 4) for h in headers]  # Increased from +2 to +4

    # Create header row with separator
    header_str = " | ".join(["{0:^{1}}".format(h, w) for h, w in zip(headers, col_widths)])
    print(Style.BRIGHT + header_str + Style.RESET_ALL)
    print("-" * len(header_str))

    # Create rows for each data type
    for i, data_type in enumerate(data_types):
        result = results_by_type[i]

        # Check if built-in is available
        if 'built_in' not in result:
            print("No built-in results for size {}".format(size))
            continue

        built_in_mean = result['built_in']['mean']
        built_in_stdev = result['built_in']['stdev']

        # Initialize arrays of values to print with corresponding colors
        values = []
        colors = []

        # Add size
        values.append(data_type.capitalize())
        colors.append(Fore.WHITE)  # Default color for size

        # Format values with standard deviations
        bi_val = "{:.4f}±{:.4f}".format(built_in_mean, built_in_stdev)

        # Add built-in as reference (not competing)
        values.append(bi_val)
        colors.append(Fore.WHITE)  # Neutral color for reference point

        # Initialize competing methods and times
        competing_methods = []
        competing_times = []

        # Add Python implementation if available
        if 'python_impl' in result:
            python_impl_mean = result['python_impl']['mean']
            python_impl_stdev = result['python_impl']['stdev']
            py_val = "{:.4f}±{:.4f}".format(python_impl_mean, python_impl_stdev)
            python_impl_ratio = built_in_mean / python_impl_mean if python_impl_mean > 0 else float('inf')

            values.append(py_val)
            competing_methods.append(('python_impl', python_impl_mean))
            competing_times.append(python_impl_mean)

            # Add performance ratio for Python implementation
            values.append("{:.2f}x".format(python_impl_ratio))

            # Color code Python ratio (higher is better)
            if python_impl_ratio > 1.1:
                colors.append(Fore.GREEN)  # Python impl is faster than built-in
                colors.append(Fore.GREEN)
            elif python_impl_ratio < 0.9:
                colors.append(Fore.RED)  # Python impl is slower than built-in
                colors.append(Fore.RED)
            else:
                colors.append(Fore.YELLOW)  # Python impl is about the same as built-in
                colors.append(Fore.YELLOW)

        # C Extension if available
        if 'c_extension' in result:
            c_ext_mean = result['c_extension']['mean']
            c_ext_stdev = result['c_extension']['stdev']
            c_ext_ratio = built_in_mean / c_ext_mean if c_ext_mean > 0 else float('inf')
            c_ext_val = "{:.4f}±{:.4f}".format(c_ext_mean, c_ext_stdev)
            competing_methods.append(('c_extension', c_ext_mean))
            competing_times.append(c_ext_mean)

            values.append(c_ext_val)
            colors.append(Fore.WHITE)  # Placeholder, will be colored based on performance later

            values.append("{:.2f}x".format(c_ext_ratio))

            # Color code C ratio compared to built-in
            if c_ext_ratio > 1.1:
                colors.append(Fore.GREEN)
            elif c_ext_ratio < 0.9:
                colors.append(Fore.RED)
            else:
                colors.append(Fore.YELLOW)

        # Cython Extension if available
        if 'cython_extension' in result:
            cython_ext_mean = result['cython_extension']['mean']
            cython_ext_stdev = result['cython_extension']['stdev']
            cython_ext_ratio = built_in_mean / cython_ext_mean if cython_ext_mean > 0 else float('inf')
            cython_ext_val = "{:.4f}±{:.4f}".format(cython_ext_mean, cython_ext_stdev)
            competing_methods.append(('cython_extension', cython_ext_mean))
            competing_times.append(cython_ext_mean)

            values.append(cython_ext_val)
            colors.append(Fore.WHITE)  # Placeholder, will be colored based on performance later

            values.append("{:.2f}x".format(cython_ext_ratio))

            # Color code Cython ratio compared to built-in
            if cython_ext_ratio > 1.1:
                colors.append(Fore.GREEN)
            elif cython_ext_ratio < 0.9:
                colors.append(Fore.RED)
            else:
                colors.append(Fore.YELLOW)

        # Apply performance-based colors for implementation times
        if competing_times:
            min_time = min(competing_times)
            max_time = max(competing_times)

            # Update colors for each competing method's time value
            time_color_indices = []
            if 'python_impl' in result:
                time_color_indices.append(2)  # Index of python_impl time in values
            if 'c_extension' in result:
                offset = 4 if 'python_impl' in result else 2
                time_color_indices.append(offset)  # Index of c_extension time in values
            if 'cython_extension' in result:
                offset = 6 if 'python_impl' in result else 4
                offset = offset - 2 if 'c_extension' not in result else offset
                time_color_indices.append(offset)  # Index of cython_extension time in values

            for idx, method_name in enumerate(method for method, _ in competing_methods):
                method_mean = result[method_name]['mean']
                if idx < len(time_color_indices):
                    color_idx = time_color_indices[idx]
                    if color_idx < len(colors):
                        # Normalize time for color coding
                        if max_time == min_time:  # Avoid division by zero
                            normalized = 0
                        else:
                            normalized = (method_mean - min_time) / (max_time - min_time)

                        if normalized < 0.2:
                            colors[color_idx] = Fore.GREEN  # Very fast
                        elif normalized < 0.4:
                            colors[color_idx] = Fore.LIGHTGREEN_EX  # Fast
                        elif normalized < 0.6:
                            colors[color_idx] = Fore.YELLOW  # Medium
                        elif normalized < 0.8:
                            colors[color_idx] = Fore.LIGHTYELLOW_EX  # Slow
                        else:
                            colors[color_idx] = Fore.RED  # Very slow

        # Determine winner (method with minimum time) - excluding built-in
        if competing_methods:
            winner_method, _ = min(competing_methods, key=lambda x: x[1])
            winner_map = {
                'python_impl': 'Python-impl',
                'c_extension': 'C Extension',
                'cython_extension': 'Cython'
            }
            winner = winner_map.get(winner_method, winner_method)
        else:
            winner = "N/A"

        values.append(winner)
        colors.append(Fore.CYAN)  # Winner color

        # Print the row with proper spacing
        parts = []
        for val, color, w in zip(values, colors, col_widths):
            parts.append(color + "{0:^{1}}".format(val, w) + Style.RESET_ALL)
        row_str = " | ".join(parts)
        print(row_str)


def plot_results(size, results_by_type, data_types):
    """Generate performance comparison plots with error bars"""
    # First plot - Absolute performance with error bars
    fig1 = plt.figure(figsize=(12, 8))

    # Extract data for plotting with error bars
    built_in_means = [r['built_in']['mean'] for r in results_by_type]
    floor_div_means = [r['floor_div']['mean'] for r in results_by_type]
    python_impl_means = [r['python_impl']['mean'] for r in results_by_type]

    built_in_errors = [r['built_in']['stdev'] for r in results_by_type]
    floor_div_errors = [r['floor_div']['stdev'] for r in results_by_type]
    python_impl_errors = [r['python_impl']['stdev'] for r in results_by_type]

    # Plot with error bars for standard deviation
    plt.errorbar(data_types, built_in_means, yerr=built_in_errors, fmt='b-o', linewidth=2, markersize=7, capsize=5,
                 label='Built-in (/)')
    plt.errorbar(data_types, floor_div_means, yerr=floor_div_errors, fmt='g-s', linewidth=2, markersize=7, capsize=5,
                 label='Floor div (//)')
    plt.errorbar(data_types, python_impl_means, yerr=python_impl_errors, fmt='r--^', linewidth=2, markersize=7, capsize=5,
                 label='Python-impl div (//)')

    # Plot C extension if available
    if has_c_extension:
        c_ext_means = [r['c_extension']['mean'] for r in results_by_type]
        c_ext_errors = [r['c_extension']['stdev'] for r in results_by_type]
        plt.errorbar(data_types, c_ext_means, yerr=c_ext_errors, fmt='-D', color='darkorange', linewidth=2, markersize=7,
                     capsize=5, label='C Extension')

    # Plot Cython extension if available
    if has_cython_extension:
        cython_ext_means = [r['cython_extension']['mean'] for r in results_by_type]
        cython_ext_errors = [r['cython_extension']['stdev'] for r in results_by_type]
        plt.errorbar(data_types, cython_ext_means, yerr=cython_ext_errors, fmt='-p', color='purple', linewidth=2,
                     markersize=7, capsize=5, label='Cython Extension')

    # Second plot - Relative performance (bar chart)
    # Calculate relative performance (normalized to built-in division)
    # built-in / implementation time - smaller is better (less time)
    built_in_relative = np.ones(len(data_types))  # baseline (1.0)
    floor_div_relative = np.array(built_in_means) / np.array(floor_div_means)
    python_impl_relative = np.array(built_in_means) / np.array(python_impl_means)

    if has_c_extension:
        c_ext_relative = np.array(built_in_means) / np.array([r['c_extension']['mean'] for r in results_by_type])

    if has_cython_extension:
        cython_ext_relative = np.array(built_in_means) / np.array(
            [r['cython_extension']['mean'] for r in results_by_type])

    ind = np.arange(len(data_types))  # x locations for the bars
    width = 0.15  # width of the bars

    # Clear the previous figure before creating a new one
    plt.close(fig1)
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Create bars
    rects1 = ax.bar(ind - 2 * width, built_in_relative, width, label='Built-in (/)')
    rects2 = ax.bar(ind - width, floor_div_relative, width, label='Floor div (//)')
    rects3 = ax.bar(ind, python_impl_relative, width, label='Python-impl')

    if has_c_extension:
        rects4 = ax.bar(ind + width, c_ext_relative, width, label='C Extension')

    if has_cython_extension:
        rects5 = ax.bar(ind + 2 * width, cython_ext_relative, width, label='Cython Extension')

    # Add labels and annotations
    def autolabel(rects):
        """Attach a text label above each bar"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}x'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Apply autolabels to all bar sets
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    if has_c_extension:
        autolabel(rects4)
    if has_cython_extension:
        autolabel(rects5)

    # Set up the plot with labels and title
    ax.set_ylabel('Performance Ratio (higher is better)', fontsize=12)
    ax.set_xlabel('Data Type', fontsize=12)
    ax.set_title('{} Division Relative Performance'.format('PY2' if PY2 else 'PY3'), fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(data_types)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)  # Reference line at y=1

    # Add value labels on top of each bar
    plt.tight_layout()
    plt.savefig('division_benchmark_relative_{}.png'.format('py2' if PY2 else 'py3'), dpi=300)
    # IMPORTANT: Show the second (relative) plot
    plt.show()
    plt.close(fig2)


def main():
    """Main function to run the benchmarks"""
    print("Python Division Benchmark")
    print("Python version: {}".format(platform.python_version()))
    print("Platform: {}".format(platform.platform()))

    # Available benchmark functions
    functions = {
        'built_in': builtin_division,
        'floor_div': floor_division,
        'python_impl': python_division
    }

    # Add C extension if available
    if has_c_extension:
        print("C extension available")
        functions['c_extension'] = c_division.div_wrapper
    else:
        print("C extension not available")

    # Add Cython extension if available
    if has_cython_extension:
        print("Cython extension available")
        functions['cython_extension'] = cy_division.div_wrapper
    else:
        print("Cython extension not available")

    # Data types to benchmark
    data_types = ['int', 'float', 'mixed', 'huge']

    # Data size to test
    size = 10000

    # Number of runs for each benchmark
    num_runs = 10

    # Run benchmarks for each data type
    results_by_types = []
    for data_type in data_types:

        # For each size, run all functions
        type_results = {}

        for name, func in functions.items():
            try:
                result = run_benchmark(func, size, data_type, num_runs)
                type_results[name] = result
            except Exception as e:
                print("Error running {} on {} data (size={}): {}".format(
                    name, data_type, size, str(e)))

        results_by_types.append(type_results)

        # Print results
    print_results_table(size, results_by_types, data_types)
    if not noPics:
        plot_results(size, results_by_types, data_types)


if __name__ == "__main__":
    main()

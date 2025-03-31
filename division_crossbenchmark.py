# -*- coding: utf-8 -*-

"""
Division Benchmark with Python 2 Data, All Comparisons & Relative Performance Graphics
---------------------------------------------------------------------------------------
Compares different division implementations across numerical types, includes:
    - Python 2-style division
    - Built-in Python division
    - Cython extension (if available)
    - C extension (if available)
    - Pure Python implementations
    - Dumping/loading of Python 2 performance data for comparison
    - Graphical plots for performance comparison
"""

from __future__ import print_function, absolute_import, division
import sys
import time
import random
import platform
import math
import json
import os

import numpy as np

try:
	import matplotlib.pyplot as plt
	# Define default Matplotlib style
	try:
		plt.style.use('ggplot')  # Use a built-in style for consistency
	except AttributeError:
		print("Style not found. Using default Matplotlib style.")

	noPics = False
except ImportError:
	noPics = True

try:
	import py2py3div_c as c_division
	has_c_extension = True
except ImportError:
	has_c_extension = False

try:
	import py2py3div_cython as cy_division
	has_cython_extension = True
except ImportError:
	has_cython_extension = False

from py2py3div_python import builtin_division, div_wrapper as python_division
# from py2py3div_python import div_wrapper as builtin_division

# Check Python version
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# File for storing Python 2 benchmark data
PY2_DATA_FILE = "py2_division_benchmark.json"

if PY2:
	# timer_impl = time.time
	timer_impl = time.clock
else:
	# timer_impl = time.process_time
	timer_impl = time.perf_counter


def run_benchmark(func, size, data_type, num_runs=5):
	"""Run a benchmark and return timing statistics."""
	if data_type == 'int':
		numerators = [random.randint(1, 10000) for _ in range(size)]
		denominators = [random.randint(1, 1000) for _ in range(size)]
	elif data_type == 'float':
		numerators = [random.random() * 10000 for _ in range(size)]
		denominators = [random.random() * 1000 + 0.001 for _ in range(size)]
	elif data_type == 'mixed':
		numerators = [
			random.choice([random.randint(1, 10000), random.random() * 10000, random.randint(1, 10000) * (10 ** 18)])
			for _ in range(size)
		]
		denominators = [random.choice([random.randint(1, 1000), random.random() * 1000 + 0.001, random.randint(1, 100) * (10 ** 18)])
						for _ in range(size)]
	elif data_type == 'huge':
		numerators = [random.randint(1, 10000) * (10 ** 18) for _ in range(size)]
		denominators = [random.randint(1, 1000) * (10 ** 18) for _ in range(size)]
	else:
		raise ValueError("Unsupported data type: {}".format(data_type))

	times = []
	for _ in range(num_runs):
		start_time = timer_impl()
		for n, d in zip(numerators, denominators):
			func(n, d)
		end_time = timer_impl()
		times.append((end_time - start_time) * 1000000000.0)

	try:
		mean_time = np.mean(times)
		stdev = np.std(times, ddof=0)  # Population standard deviation
	except:
		mean_time = sum(times) / len(times)
		stdev = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times)) if len(times) > 1 else 0

	return {'mean': mean_time / float(size), 'stdev': stdev / float(size)}

def dump_py2_data(results_by_type, data_types):
	"""Dump Python 2 benchmarking data to a file."""
	try:
		with open(PY2_DATA_FILE, 'w') as f:
			json.dump({'data_types': data_types, 'results_by_type': results_by_type}, f, indent=2, sort_keys=True)
		print("Python 2 benchmark data has been successfully saved to '{}'.".format(PY2_DATA_FILE))
	except Exception as e:
		print("Failed to save benchmark data: {}".format(str(e)))

def load_py2_data():
	"""Load Python 2 benchmarking data from file."""
	if not os.path.exists(PY2_DATA_FILE):
		print("No Python 2 benchmark data file '{}' found.".format(PY2_DATA_FILE))
		return None
	try:
		with open(PY2_DATA_FILE, 'r') as f:
			data = json.load(f)
		print("Python 2 benchmark data has been successfully loaded from '{}'.".format(PY2_DATA_FILE))
		return data
	except Exception as e:
		print("Failed to load benchmark data: {}".format(str(e)))
		return None

def print_results_table(results_by_type, data_types, py2_data=None):
	"""Print benchmark results in a well-formatted table with color-coding and Python 2 data from JSON"""
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

	# Add Python 2 columns if data is available
	if py2_data:
		headers.extend(["Python-2", "Py2/Built-in"])

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
		bi_val = "{:.2f}±{:.2f}".format(built_in_mean, built_in_stdev)

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
			py_val = "{:.2f}±{:.2f}".format(python_impl_mean, python_impl_stdev)
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
			c_ext_val = "{:.2f}±{:.2f}".format(c_ext_mean, c_ext_stdev)
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
			cython_ext_val = "{:.2f}±{:.2f}".format(cython_ext_mean, cython_ext_stdev)
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

		# Add Python 2 data if available
		if py2_data and i < len(py2_data['results_by_type']):
			py2_result = py2_data['results_by_type'][i]['built_in']
			py2_mean = py2_result['mean']
			py2_stdev = py2_result['stdev']
			py2_ratio = built_in_mean / py2_mean if py2_mean > 0 else float('inf')
			py2_val = "{:.3f}±{:.3f}".format(py2_mean, py2_stdev)

			# Add to the competing methods
			competing_methods.append(('python_2', py2_mean))
			competing_times.append(py2_mean)

			values.append(py2_val)
			colors.append(Fore.WHITE)  # Placeholder, will be colored based on performance later

			values.append("{:.2f}x".format(py2_ratio))

			# Color code Python 2 ratio compared to built-in
			if py2_ratio > 1.1:
				colors.append(Fore.GREEN)
			elif py2_ratio < 0.9:
				colors.append(Fore.RED)
			else:
				colors.append(Fore.YELLOW)

		# Apply performance-based colors for implementation times
		if competing_times:
			min_time = min(competing_times)
			max_time = max(competing_times)

			# Update colors for each competing method's time value
			time_color_indices = []

			# Calculate the indices of time values in the values array
			current_idx = 2
			if 'python_impl' in result:
				time_color_indices.append(current_idx)
				current_idx += 2

			if 'c_extension' in result:
				time_color_indices.append(current_idx)
				current_idx += 2

			if 'cython_extension' in result:
				time_color_indices.append(current_idx)
				current_idx += 2

			if py2_data and i < len(py2_data['results_by_type']):
				time_color_indices.append(current_idx)
				current_idx += 2

			for idx, method_name in enumerate(method for method, _ in competing_methods):
				method_mean = competing_methods[idx][1]  # Get the mean execution time
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
				'cython_extension': 'Cython',
				'python_2': 'Python 2'
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


def plot_benchmark_results(results_by_type, data_types, py2_data=None):
	"""
    Enhanced version of the benchmark results plot with improved label handling and accuracy.
    """
	colors = {
		'built_in': '#1f77b4',  # blue
		'python_impl': '#ff7f0e',  # orange
		'c_extension': '#2ca02c',  # green
		'cython_extension': '#d62728',  # red
		'python_2': '#9467bd'  # purple
	}

	plt.figure(figsize=(12, 8), dpi=100)

	x = np.arange(len(data_types))
	bar_width = 0.15
	opacity = 0.9

	built_in_times = []
	python_impl_times = []
	c_extension_times = []
	cython_extension_times = []
	python_2_times = []

	for i, result in enumerate(results_by_type):
		built_in_times.append(result['built_in']['mean'] if 'built_in' in result else 0)
		python_impl_times.append(result['python_impl']['mean'] if 'python_impl' in result else 0)
		c_extension_times.append(result['c_extension']['mean'] if 'c_extension' in result else 0)
		cython_extension_times.append(result['cython_extension']['mean'] if 'cython_extension' in result else 0)

		if py2_data and i < len(py2_data['results_by_type']):
			python_2_data = py2_data['results_by_type'][i]
			python_2_times.append(python_2_data['built_in']['mean'] if 'built_in' in python_2_data else 0)
		else:
			python_2_times.append(0)

	bar_offset = -bar_width * 2

	if built_in_times:
		plt.bar(x + bar_offset, built_in_times, bar_width, alpha=opacity, color=colors['built_in'], label='Built-in')
		bar_offset += bar_width

	if python_impl_times:
		plt.bar(x + bar_offset, python_impl_times, bar_width, alpha=opacity, color=colors['python_impl'],
				label='Python-impl')
		bar_offset += bar_width

	if c_extension_times:
		plt.bar(x + bar_offset, c_extension_times, bar_width, alpha=opacity, color=colors['c_extension'],
				label='C-Extension')
		bar_offset += bar_width

	if cython_extension_times:
		plt.bar(x + bar_offset, cython_extension_times, bar_width, alpha=opacity, color=colors['cython_extension'],
				label='Cython-Extension')
		bar_offset += bar_width

	if python_2_times:
		plt.bar(x + bar_offset, python_2_times, bar_width, alpha=opacity, color=colors['python_2'], label='Python 2')

	# Add value labels to bars with dynamic decimal formatting
	def autolabel(heights, positions):
		for x_pos, h in zip(positions, heights):
			if h > 0:
				label = '{:.2f}'.format(h)  # Two decimals for large values

				plt.text(
					x_pos, h + (max(heights) * 0.01), label, fontsize=8,
					ha='center', va='bottom', color='black', alpha=0.8
				)

	autolabel(built_in_times, x - 0.3)
	autolabel(python_impl_times, x - 0.15)
	autolabel(c_extension_times, x + 0.0)
	autolabel(cython_extension_times, x + 0.15)
	autolabel(python_2_times, x + 0.3)

	plt.title('Division Benchmark Results ({})'.format('PY2' if PY2 else 'PY3'), fontsize=16, fontweight='bold')
	plt.xlabel('Data Type', fontsize=12, labelpad=15)
	plt.ylabel('Execution Time (ns)', fontsize=12, labelpad=15)
	plt.xticks(x, [dtype.capitalize() for dtype in data_types], fontsize=10, rotation=25)
	plt.legend(title='Implementation', loc='upper left', fontsize=10, frameon=True)
	plt.grid(visible=True, linestyle='--', alpha=0.6)

	plt.tight_layout()
	plt.savefig('plots/division_benchmark_{}.png'.format('py2' if PY2 else 'py3'), dpi=300, bbox_inches='tight')
	plt.show()

	# # Auto-scale the y-axis
	# max_value = max(max(vals) for vals in relative_performance.values() if vals)
	# plt.ylim(0, max_value * 1.2)  # Add 20% headroom


def plot_relative_performance(results_by_type, data_types, py2_data=None):
	"""
	Generate relative performance comparison plots across all methods.
	Shows each method's performance relative to the built-in division.
	"""
	# Define colors for different implementations
	colors = {
		'python_impl': '#ff7f0e',  # orange
		'c_extension': '#2ca02c',  # green
		'cython_extension': '#d62728',  # red
		'python_2': '#9467bd',  # purple
		'built_in': '#1f77b4'  # blue (reference)
	}

	# Extract built-in means as the baseline
	built_in_means = [r['built_in']['mean'] for r in results_by_type]

	# Collect all available methods
	methods = []
	if any('python_impl' in r for r in results_by_type):
		methods.append('python_impl')
	if has_c_extension and any('c_extension' in r for r in results_by_type):
		methods.append('c_extension')
	if has_cython_extension and any('cython_extension' in r for r in results_by_type):
		methods.append('cython_extension')

	# Add Python 2 data if available
	py2_available = False
	if py2_data and 'results_by_type' in py2_data:
		py2_available = True
		methods.append('python_2')

	# Calculate relative performance (built-in / method)
	relative_performance = {}

	# Built-in is always 1.0 (reference)
	relative_performance['built_in'] = [1.0] * len(data_types)

	# Calculate each method's relative performance
	for method in methods:
		if method == 'python_2':
			# Python 2 data comes from the separate JSON file
			if py2_data and 'results_by_type' in py2_data:
				py2_means = [r['built_in']['mean'] for r in py2_data['results_by_type']]
				relative_performance[method] = [b/p if p > 0 else float('inf')
												for b, p in zip(built_in_means, py2_means)]
		else:
			# Other methods come from the current benchmark
			method_means = [r[method]['mean'] if method in r else float('inf')
							for r in results_by_type]
			relative_performance[method] = [b/m if m > 0 else float('inf')
											for b, m in zip(built_in_means, method_means)]

	# Set up the figure with proper size and DPI
	plt.figure(figsize=(12, 8), dpi=100)

	# Set up the x positions
	x = np.arange(len(data_types))
	bar_width = 0.18

	# Calculate total methods including built-in reference
	total_methods = len(methods) + 1

	# Calculate starting position to center the group of bars
	start_x = -(total_methods * bar_width) / 2 + (bar_width / 2)

	# Plot built-in reference bars (always 1.0)
	rects_built_in = plt.bar(x + start_x,
							 relative_performance['built_in'],
							 bar_width,
							 color=colors['built_in'],
							 label='Built-in (reference)')

	# Plot other methods' relative performance
	all_rects = [rects_built_in]
	for i, method in enumerate(methods):
		method_rects = plt.bar(x + start_x + (i + 1) * bar_width,
							   relative_performance[method],
							   bar_width,
							   color=colors[method],
							   label=method.replace('_', '-').title())
		all_rects.append(method_rects)

	# Annotate bars with their relative performance values
	def autolabel(rects, values):
		for rect, val in zip(rects, values):
			height = rect.get_height()
			plt.annotate('{:.2f}x'.format(val),
						 xy=(rect.get_x() + rect.get_width() / 2, height),
						 xytext=(0, 3),  # 3 points vertical offset
						 textcoords="offset points",
						 ha='center', va='bottom',
						 fontsize=9)

	# Add annotations to all bars
	for rects, method in zip(all_rects, ['built_in'] + methods):
		autolabel(rects, relative_performance[method])

	# Add labels, title and customize the chart
	plt.xlabel('Data Type', fontsize=12)
	plt.ylabel('Relative Performance (higher is better)', fontsize=12)
	plt.title('Division Methods: Performance Relative to Built-in ({})'.format('py2' if PY2 else 'py3'), fontsize=16, fontweight='bold')
	plt.xticks(x, [dtype.capitalize() for dtype in data_types], fontsize=10)
	plt.legend(title='Implementation', loc='upper left', fontsize=10, frameon=True)
	plt.grid(True, linestyle='--', alpha=0.7, axis='y')

	# Auto-scale the y-axis
	max_value = max(max(vals) for vals in relative_performance.values() if vals)
	plt.ylim(0, max_value * 1.2)  # Add 20% headroom

	# Tight layout for better spacing
	plt.tight_layout()
	plt.savefig('plots/division_relative_performance_comparison_{}.png'.format('py2' if PY2 else 'py3'), dpi=300)
	plt.show()

def main():
	"""Main function to run the benchmarks."""
	print("Python Division Benchmark with All Comparisons and Graphics")
	print("Python version: {}".format(platform.python_version()))
	print("Platform: {}".format(platform.platform()))

	functions = {
		'built_in': builtin_division,
		'python_impl': python_division
	}
	if has_c_extension:
		functions['c_extension'] = c_division.div_wrapper
	if has_cython_extension:
		functions['cython_extension'] = cy_division.div_wrapper

	data_types = ['int', 'float', 'mixed', 'huge']
	sizes = [1000000, 1000000, 3000, 1000]
	num_runs = 100

	results_by_types = []
	for data_type in data_types:
		type_results = {}
		for function_desc, size in zip(functions.items(), sizes):
			name, func = function_desc
			type_results[name] = run_benchmark(func, size, data_type, num_runs)
		results_by_types.append(type_results)

	py2_data = None
	if PY3:
		py2_data = load_py2_data()
	elif PY2:
		dump_py2_data(results_by_types, data_types)

	print_results_table(results_by_types, data_types, py2_data)
	if not noPics:
		plot_benchmark_results(results_by_types, data_types, py2_data)
		plot_relative_performance(results_by_types, data_types, py2_data)

if __name__ == "__main__":
	main()
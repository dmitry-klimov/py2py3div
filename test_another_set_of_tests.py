from __future__ import absolute_import
import sys
import pytest
import time

# For Python 3 compatibility
if sys.version_info[0] >= 3:
    long = int
    PY_VERSION = 3
else:
    PY_VERSION = 2
    
# Assuming implementations are accessible from different modules
from py2py3div_python import div_wrapper as py2py3div_python
from py2py3div_c import div_wrapper as py2py3div_c
from py2py3div_cython import div_wrapper as py2py3div_cython

# Create a mapping of implementation names to their respective functions
IMPLEMENTATIONS = {
    'Built-In': lambda x, y: x / y,
    "Python": py2py3div_python,
    "C": py2py3div_c,
    "Cython": py2py3div_cython,
}


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_integer_division(implementation_name, div_wrapper):
    """Test integer division behaves consistently (Python 2-style)"""
    assert div_wrapper(5, 2) == 2
    assert div_wrapper(10, 3) == 3
    assert div_wrapper(1, 2) == 0
    assert div_wrapper(-5, 2) == -3
    assert div_wrapper(5, -2) == -3
    assert div_wrapper(-5, -2) == 2


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_float_division(implementation_name, div_wrapper):
    """Test float division behaves the same in Python 2 and 3"""
    assert div_wrapper(5.0, 2) == 2.5
    assert div_wrapper(5, 2.0) == 2.5
    assert div_wrapper(5.0, 2.0) == 2.5
    assert div_wrapper(-5.0, 2) == -2.5
    assert div_wrapper(5.0, -2) == -2.5
    assert div_wrapper(-5.0, -2) == 2.5


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_zero_division(implementation_name, div_wrapper):
    """Test division by zero raises appropriate exception"""
    with pytest.raises(ZeroDivisionError):
        div_wrapper(1, 0)
    with pytest.raises(ZeroDivisionError):
        div_wrapper(0, 0)
    assert div_wrapper(0, 5) == 0
    assert div_wrapper(0, -5) == 0


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_long_integers(implementation_name, div_wrapper):
    """Test division with long integers"""
    assert div_wrapper(long(5), 2) == long(2)
    assert div_wrapper(5, long(2)) == long(2)
    assert div_wrapper(long(5), long(2)) == long(2)
    large_int1 = 10 ** 100 if sys.version_info[0] >= 3 else long(10 ** 100)
    large_int2 = 10 ** 98 if sys.version_info[0] >= 3 else long(10 ** 98)
    assert div_wrapper(large_int1, large_int2) == 100


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_mixed_types(implementation_name, div_wrapper):
    """Test division with mixed numeric types"""
    assert div_wrapper(5, 2.0) == 2.5
    assert div_wrapper(5.0, 2) == 2.5
    try:
        result = div_wrapper(5, complex(2, 0))
        assert result == complex(2.5, 0)
    except (TypeError, ValueError):
        pass
    try:
        result = div_wrapper(5, (2 + 0j))
        assert result == (2.5 + 0j)
    except (TypeError, ValueError):
        pass


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_custom_classes(implementation_name, div_wrapper):
    """Test division with custom classes implementing __div__ and __rdiv__"""

    class CustomDividend:
        def __truediv__(self, other):  # Python 3
            return "truediv_result"

        def __div__(self, other):  # Python 2
            return "div_result"

    class CustomDivisor:
        def __rtruediv__(self, other):  # Python 3
            return "rtruediv_result"

        def __rdiv__(self, other):  # Python 2
            return "rdiv_result"

    dividend = CustomDividend()
    expected_result = "div_result" if sys.version_info[0] < 3 else "truediv_result"
    assert div_wrapper(dividend, 2) == expected_result
    divisor = CustomDivisor()
    expected_result = "rdiv_result" if sys.version_info[0] < 3 else "rtruediv_result"
    assert div_wrapper(5, divisor) == expected_result


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_precision(implementation_name, div_wrapper):
    """Test division precision with floating point numbers"""
    assert abs(div_wrapper(1.0, 3.0) - 0.3333333333333333) < 1e-15
    small_num = 1e-15
    large_num = 1e15
    assert abs(div_wrapper(small_num, 1.0) - small_num) < 1e-30
    assert abs(div_wrapper(1.0, large_num) - (1.0 / large_num)) < 1e-30


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_type_preservation(implementation_name, div_wrapper):
    """Test that result types are appropriate"""
    result = div_wrapper(6, 3)
    assert isinstance(result, int)
    result = div_wrapper(6.0, 3)
    assert isinstance(result, float)
    result = div_wrapper(6, 3.0)
    assert isinstance(result, float)


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_consistency_with_operators(implementation_name, div_wrapper):
    """Test consistency with standard operators in respective Python versions"""
    if sys.version_info[0] < 3:
        assert div_wrapper(5, 2) == 5 / 2
        assert div_wrapper(-5, 2) == -5 / 2
    else:
        assert div_wrapper(5, 2) == 5 // 2
        assert div_wrapper(-5, 2) == -5 // 2
    assert div_wrapper(5.0, 2) == 5.0 / 2
    assert div_wrapper(5, 2.0) == 5 / 2.0


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_special_cases(implementation_name, div_wrapper):
    """Test special numerical cases"""
    assert div_wrapper(1, 10 ** 6) == 0
    assert div_wrapper(10 ** 6, 10 ** 12) == 0
    assert abs(div_wrapper(1.0, 10 ** 6) - 1e-6) < 1e-15


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_boolean_operands(implementation_name, div_wrapper):
    """Test division with boolean operands"""
    assert div_wrapper(True, 2) == 0
    assert div_wrapper(True, 1) == 1
    assert div_wrapper(10, True) == 10
    with pytest.raises(ZeroDivisionError):
        div_wrapper(5, False)


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
@pytest.mark.parametrize("large_num1, large_num2, iterations, max_time_diff", [
    (10, 5, 10000000, 3),
    (10 ** 10, 5, 5000000, 3),
    (10 ** 100, 5, 1000000, 3)
])
def test_performance(implementation_name, div_wrapper, large_num1, large_num2, iterations, max_time_diff):
    """Test performance of div_wrapper with large inputs and compare with Python's division"""
    start_time = time.time()
    for _ in range(iterations):
        div_wrapper(large_num1, large_num2)
    python_start_time = wrapper_end_time = time.time()
    div_wrapper_elapsed = wrapper_end_time - start_time

    for _ in range(iterations):
        large_num1 / large_num2
    python_end_time = time.time()
    python_elapsed = python_end_time - python_start_time

    assert div_wrapper_elapsed <= python_elapsed * max_time_diff, (
        "div_wrapper is significantly slower. div_wrapper: {}, Python: {}".format(
            div_wrapper_elapsed, python_elapsed
        )
    )
    assert div_wrapper(large_num1, large_num2) == (large_num1 // large_num2), (
        "div_wrapper result is inconsistent with Python's division result")
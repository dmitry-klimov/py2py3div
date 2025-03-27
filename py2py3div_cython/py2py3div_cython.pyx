# cython: language_level=2
from cpython.long cimport PyLong_AsLongLong, PyLong_FromLong
from cpython.object cimport PyObject
import sys
cimport cython

# For Python 2.7 compatibility
IS_PY2 = True  # More meaningful name for the constant

# Handle int64_t for MSVC 2008
cdef extern from *:
    """
    #ifdef _MSC_VER
      #if _MSC_VER < 1600
        typedef __int64 int64_t;
      #else
        #include <stdint.h>
      #endif
    #else
      #include <stdint.h>
    #endif
    """
    ctypedef long long int64_t

# Division operations for different numeric types
cdef inline int64_t fast_int_div(int64_t a, int64_t b) nogil:
    """Integer division for Python 2.7 (GIL-free)"""
    return a // b

cdef inline double fast_float_div(double a, double b) nogil:
    """Floating-point division (GIL-free)"""
    return a / b

cdef inline object fast_bigint_div(object a, object b):
    """Big integer division using Python's native long type"""
    return a // b

# Helper methods for division protocol
cdef inline object try_division_method(object a, object b, str method_name):
    """Try to use a division method and return the result if successful"""
    if hasattr(a, method_name):
        result = getattr(a, method_name)(b)
        if result is not NotImplemented:
            return result
    return NotImplemented

cdef inline object try_object_division(object a, object b):
    """Try object division using Python's special methods"""
    # Try primary methods first
    result = try_division_method(a, "__div__", b)
    if result is not NotImplemented:
        return result

    result = try_division_method(b, "__rdiv__", a)
    if result is not NotImplemented:
        return result

    # Try fallback methods
    result = try_division_method(b, "__div__", a)
    if result is not NotImplemented:
        return result

    result = try_division_method(a, "__rdiv__", b)
    if result is not NotImplemented:
        return result

    return NotImplemented

cpdef object div_wrapper_cython(object a, object b):
    """
    Universal division function for Python 2.7.
    
    Handles integer, floating point, and object division according to Python 2.7 rules.
    """
    # Check for division by zero
    if b == 0 or b is None:
        raise ZeroDivisionError("Division by zero")

    # Handle numeric division
    if isinstance(a, (int, long)) and isinstance(b, (int, long)):
        try:
            # Try fast integer division
            return fast_int_div(PyLong_AsLongLong(a), PyLong_AsLongLong(b))
        except OverflowError:
            # Fall back to Python's big integer handling
            return fast_bigint_div(a, b)

    # Handle float division
    if isinstance(a, float) or isinstance(b, float):
        return fast_float_div(<double> a, <double> b)

    # Handle object division
    result = try_object_division(a, b)
    if result is not NotImplemented:
        return result

    # Final fallback
    try:
        return a / b
    except TypeError:
        raise TypeError(f"unsupported operand type(s) for /: '{type(a).__name__}' and '{type(b).__name__}'")
import sys

# Check Python version
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY2:
    def div_wrapper(a, b):
        """Pure Python implementation of division - Python2 version"""
        return a / b
elif PY3:
    def div_wrapper(a, b):
        """Pure Python implementation of division - Python3 version"""
        # for _ in range(10):
        #     a * b + a / b

        if isinstance(a, int) and isinstance(b, int):
            return a // b
        else:
            return a / b
else:
    # Define a fallback for debugging purposes
    def div_wrapper(*_):
        raise ImportError("Could not load Pure Python implementation")

def builtin_division(a, b):
    return a / b


# Make sure to explicitly export the function
__all__ = ['div_wrapper', 'builtin_division']
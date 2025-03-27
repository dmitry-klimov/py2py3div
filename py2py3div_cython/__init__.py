from __future__ import absolute_import  # Important for Python 2

import sys

print("Running on Python version:", sys.version)

try:
    from .py2py3div_cython import div_wrapper_cython as div_wrapper

    print("Successfully imported .div_wrapper_cython")
except ImportError as e:
    sys.stderr.write("Error importing div_wrapper_cython: %s\n" % e)

    print("Python version: %s" % sys.version)
    print("Path: %s" % sys.path)

    # Define a fallback for debugging purposes
    def div_wrapper(*_):
        raise ImportError("Could not load Cython implementation")


# Make sure to explicitly export the function
__all__ = ['div_wrapper']
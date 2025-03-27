#include <Python.h>
#include <cmath>

// Handle Python 2 vs 3 differences
#if PY_MAJOR_VERSION >= 3
  #define IS_PY3 1
  #define PyInt_Check PyLong_Check
  #define PyInt_AsLong PyLong_AsLong
  #define PyInt_FromLong PyLong_FromLong
#else
  #define IS_PY3 0
  #define PyLong_AsLongLongAndOverflow(obj, overflow) PyLong_AsLongLong(obj)
#endif

// Forward declaration of our wrapper function
static PyObject* div_wrapper_cpp(PyObject* self, PyObject* args);

// Fast floor division with optimized implementation for common cases
static inline long long fast_int_div(long long a, long long b) {
    // Quick path for positive operands (most common case)
    if (a >= 0 && b > 0) {
        return a / b;
    }

    // Quick check for when truncation and floor division are the same
    if (a % b == 0) {
        return a / b;
    }

    // For all other cases, implement floor division
    long long q = a / b;
    // Adjust if necessary to round toward negative infinity
    if ((a < 0) != (b < 0) && a % b != 0) {
        q -= 1;
    }
    return q;
}

// Main division wrapper function - optimized version
static PyObject* div_wrapper_cpp(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;

    // Parse arguments from Python
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    // Quick check for division by zero
    if (PyObject_Not(b_obj)) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Division by zero");
        return NULL;
    }

    // Fast path for integers (most common case)
    if ((PyInt_Check(a_obj) || PyLong_Check(a_obj)) &&
        (PyInt_Check(b_obj) || PyLong_Check(b_obj))) {

        // First try with longs since they're likely most common
        if (PyInt_Check(a_obj) && PyInt_Check(b_obj)) {
            long a_int = PyInt_AsLong(a_obj);
            long b_int = PyInt_AsLong(b_obj);

            if (!PyErr_Occurred()) {
                return PyInt_FromLong((long)fast_int_div(a_int, b_int)); // Cast to long to avoid warning
            }
            PyErr_Clear();
        }

        // Try with long long for larger integers
        int overflow_a = 0, overflow_b = 0;
        long long a_ll = 0, b_ll = 0;

        #if PY_MAJOR_VERSION >= 3
        a_ll = PyLong_AsLongLongAndOverflow(a_obj, &overflow_a);
        b_ll = PyLong_AsLongLongAndOverflow(b_obj, &overflow_b);
        #else
        // Python 2 - just try the conversion
        a_ll = PyLong_AsLongLong(a_obj);
        overflow_a = PyErr_Occurred() != NULL;
        if (overflow_a) PyErr_Clear();

        b_ll = PyLong_AsLongLong(b_obj);
        overflow_b = PyErr_Occurred() != NULL;
        if (overflow_b) PyErr_Clear();
        #endif

        if (!overflow_a && !overflow_b && !PyErr_Occurred()) {
            return PyLong_FromLongLong(fast_int_div(a_ll, b_ll));
        }

        // Clear errors from conversions
        if (PyErr_Occurred()) PyErr_Clear();

        // Fall back to Python's standard division for very large integers
        #if PY_MAJOR_VERSION >= 3
        return PyNumber_FloorDivide(a_obj, b_obj);
        #else
        return PyNumber_Divide(a_obj, b_obj);
        #endif
    }

    // Fast path for floats
    if (PyFloat_Check(a_obj) || PyFloat_Check(b_obj)) {
        double a_float = PyFloat_AsDouble(a_obj);
        double b_float = PyFloat_AsDouble(b_obj);

        if (!PyErr_Occurred()) {
            return PyFloat_FromDouble(a_float / b_float);
        }
        PyErr_Clear();
    }

    // Dispatch to appropriate Python methods based on version
    PyObject* result = NULL;

    #if PY_MAJOR_VERSION >= 3
    // In Python 3, we use floor division to match Python 2's behavior
    result = PyNumber_FloorDivide(a_obj, b_obj);
    #else
    // In Python 2, just use regular division
    result = PyNumber_Divide(a_obj, b_obj);
    #endif

    if (result) {
        return result;
    }

    // If we reach here, the operation failed
    PyErr_Format(PyExc_TypeError,
                "unsupported operand type(s) for /: '%s' and '%s'",
                Py_TYPE(a_obj)->tp_name, Py_TYPE(b_obj)->tp_name);
    return NULL;
}

// Method definitions
static PyMethodDef PyDivMethods[] = {
    {"div_wrapper", div_wrapper_cpp, METH_VARARGS, "Fast division compatible with Python 2/3"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

#if PY_MAJOR_VERSION >= 3
    // Python 3 module definition
    static struct PyModuleDef py2py3div_c_module = {
        PyModuleDef_HEAD_INIT,
        "py2py3div_c",
        "Python 2/3 compatible division module",
        -1,
        PyDivMethods
    };

    // Python 3 initialization function
    PyMODINIT_FUNC PyInit_py2py3div_c(void) {
        return PyModule_Create(&py2py3div_c_module);
    }
#else
    // Python 2 initialization function
    PyMODINIT_FUNC initpy2py3div_c(void) {
        Py_InitModule3("py2py3div_c", PyDivMethods, "Python 2/3 compatible division module");
    }
#endif
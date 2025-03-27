# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import pytest

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
    "Python": py2py3div_python,
    "C": py2py3div_c,
    "Cython": py2py3div_cython,
}


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_division_operator(implementation_name, div_wrapper):
    # Деление целых чисел
    assert div_wrapper(5, 2) == 5 // 2  # целочисленное деление
    assert div_wrapper(-5, 2) == -5 // 2  # округление вниз для отрицательных чисел
    # Деление с плавающей точкой
    assert div_wrapper(5.0, 2) == 5.0 / 2  # обычное деление
    assert div_wrapper(-5.0, 2) == -5.0 / 2
    # Деление нуля
    assert div_wrapper(0, 5) == 0 / 5
    assert div_wrapper(0, -5) == 0 / -5
    # Деление на единицу
    assert div_wrapper(10, 1) == 10 / 1
    assert div_wrapper(-10, 1) == -10 / 1


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_edge_cases(implementation_name, div_wrapper):
    # Деление на ноль (должно вызывать исключение)
    try:
        div_wrapper(1, 0)
        assert False, "Expected ZeroDivisionError"
    except ZeroDivisionError:
        pass
    # Деление очень больших чисел
    assert div_wrapper(1e100, 1e50) == 1e100 / 1e50
    assert div_wrapper(-1e100, 1e50) == -1e100 / 1e50
    assert div_wrapper(1e100, 1e50) == 1e50
    assert div_wrapper(-1e100, 1e50) == -1e50
    # Деление очень маленьких чисел
    assert div_wrapper(1e-100, 1e-50) == 1e-100 / 1e-50
    assert div_wrapper(-1e-100, 1e-50) == -1e-100 / 1e-50
    assert div_wrapper(1e-100, 1e-50) == 1e-50
    assert div_wrapper(-1e-100, 1e-50) == -1e-50


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_mixed_types(implementation_name, div_wrapper):
    # Деление целого на вещественное
    assert div_wrapper(5, 2.0) == 5 / 2.0
    assert div_wrapper(-5, 2.0) == -5 / 2.0
    # Деление вещественного на целое
    assert div_wrapper(5.0, 2) == 5.0 / 2
    assert div_wrapper(-5.0, 2) == -5.0 / 2
    # Деление long и int
    assert div_wrapper(long(5), 2) == long(5) // 2  # long / int → long
    assert div_wrapper(5, long(2)) == 5 // long(2)  # int / long → long
    # Деление с комплексными числами (не поддерживается)
    try:
        div_wrapper(5, (2 + 0j))
        assert False, "Expected TypeError"
    except TypeError:
        pass


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_custom_classes(implementation_name, div_wrapper):
    class MyNumber(object):
        def __div__(self, other):
            return 42

    n = MyNumber()
    assert div_wrapper(n, 5) == 42  # проверка метода __div__
    assert div_wrapper(n, "anything") == 42

    class MyNumber2(object):
        def __rdiv__(self, other):
            return 24

    n2 = MyNumber2()
    assert div_wrapper(5, n2) == 24  # проверка метода __rdiv__


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_performance(implementation_name, div_wrapper):
    import time
    start = time.time()
    for _ in range(100000000):
        div_wrapper(1, 1)
    end = time.time()
    assert end - start < 30.0, "Division operation too slow"


# def test_with_future_division():
#     from __future__ import division
#     # С future import поведение меняется на Python 3-style
#     assert div_wrapper_cython(5, 2) == 5 / 2.0  # true division
#     assert 5 // 2 == 2  # floor division (этот тест оставлен с // для демонстрации)
#     # Проверка, что другие операции не сломались
#     assert div_wrapper_cython(5.0, 2) == 5.0 / 2
#     assert div_wrapper_cython(-5, 2) == -5 / 2.0


@pytest.mark.parametrize("implementation_name, div_wrapper", IMPLEMENTATIONS.items())
def test_asymmetric_operands(implementation_name, div_wrapper):
    # Деление разных типов
    assert div_wrapper(10, 3) == 10 // 3
    assert div_wrapper(10.0, 3) == 10.0 / 3
    assert div_wrapper(10, 3.0) == 10 / 3.0
    assert div_wrapper(10.0, 3.0) == 10.0 / 3.0
    # Деление с long
    assert div_wrapper(10, long(3)) == 10 // long(3)
    assert div_wrapper(long(10), 3) == long(10) // 3
    assert div_wrapper(long(10), long(3)) == long(10) // long(3)

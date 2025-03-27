import sys

from setuptools import setup, Extension

# Define optimization flags based on compiler
if sys.platform == 'win32':
    # Windows optimization flags (MSVC)
    extra_compile_args = ['/O2', '/Ot', '/GL', '/arch:AVX2']
    extra_link_args = ['/LTCG']
else:
    # Unix-like systems (GCC/Clang)
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-flto']
    extra_link_args = ['-flto']

if sys.maxsize > 2 ** 32:  # 64-bit Python
    extra_link_args.append('/MACHINE:X64')
else:  # 32-bit Python
    extra_link_args.append('/MACHINE:X86')


# Configure the Cython extension
extension = Extension(
    'py2py3div_cython',
    sources=['py2py3div_cython.pyx'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[],
    libraries=[]
)

# Setup parameters
setup(
    name='py2py3div_cython',
    version='0.1.0',
    description='Python 2/3 compatible division module',
    ext_modules=[extension],
    setup_requires=['setuptools>=18.0', 'cython>=0.28.5'],
)

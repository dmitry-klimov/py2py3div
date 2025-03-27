from setuptools import setup, Extension
import sys


# Define linker flags based on the system architecture
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


# Define the extension module
module = Extension(
    'py2py3div_c',
    sources=['py2py3div_c.cpp'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[],
    library_dirs=[],
    libraries=[],
)

# Setup parameters
setup(
    name='py2py3div_c',
    version='0.1.0',
    description='Python 2/3 compatible division module',
    ext_modules=[module]
)
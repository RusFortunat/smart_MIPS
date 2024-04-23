from distutils.core import setup
from Cython.Build import cythonize

setup(name="environment", ext_modules=cythonize('environment.pyx'))
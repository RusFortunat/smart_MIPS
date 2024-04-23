from distutils.core import setup
from Cython.Build import cythonize

setup(name="fastllf", ext_modules=cythonize('fastllf.pyx'))
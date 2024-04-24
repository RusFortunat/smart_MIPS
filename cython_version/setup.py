from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(name=f'environment',
              sources=[f'environment.pyx'],
              language='c++',
              extra_compile_args=['-std=c++11', '-O3'],
              ),
]

setup(name=__name__,
      ext_modules=cythonize(extensions),
      )
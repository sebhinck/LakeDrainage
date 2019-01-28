from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os
from glob import glob

extra_compile_args=["-O3", "-ffast-math"]

sources = ['cython/LakeDrainageModel.pyx']
#sources += glob('c++/LakeDrainageModel.cc')
sources += glob('c++/*.cc')
inc_dirs = [numpy.get_include()]
inc_dirs += ['c++/']

# Define the extension
extension = Extension("LakeDrainage",
                      sources=sources,
                      include_dirs=inc_dirs,
                      extra_compile_args=extra_compile_args,
                      language="c++")

setup(
    name = "LakeDrainageModel",
    version = "0.0.1",
    description = "Lake Drainage Model",
    long_description = """Find drainage route of lakes""",
    author = "Sebastian Hinck",
    author_email = "sebastian.hinck@awi.de",
    url = "...",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [extension]
)


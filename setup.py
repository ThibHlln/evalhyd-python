import sys
import os

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import numpy


__version__ = '0.0.1'


ext_modules = [
    Pybind11Extension(
        "evalhyd",
        ['src/evalhyd-python.cpp',
         'deps/evalhyd/src/probabilist/evaluator_brier.cpp',
         'deps/evalhyd/src/probabilist/evaluator_elements.cpp',
         'deps/evalhyd/src/probabilist/evaluator_quantiles.cpp'],
        include_dirs=[
            numpy.get_include(),
            os.path.join(os.getcwd(), 'deps', 'evalhyd', 'deps', 'xtl',
                         'include'),
            os.path.join(os.getcwd(), 'deps', 'evalhyd', 'deps', 'xtensor',
                         'include'),
            os.path.join(os.getcwd(), 'deps', 'xtensor-python', 'include'),
            os.path.join(os.getcwd(), 'deps', 'evalhyd', 'include'),
            os.path.join(os.getcwd(), 'deps', 'evalhyd', 'src'),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include')
        ],
        language='c++',
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name='evalhyd-python',
    version=__version__,
    author='Thibault Hallouin',
    author_email='thibault.hallouin@inrae.fr',
    url='https://gitlab.irstea.fr/hycar-hydro/evalhyd/evalhyd-python',
    description='Python bindings for EvalHyd',
    long_description='An evaluator for streamflow predictions.',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)

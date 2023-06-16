import sys
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import numpy


# collect centrally sourced package version
with open("evalhyd/version.py", 'r') as fv:
    exec(fv.read())

# vendor dependencies (unless told otherwise via environment variable)
deps = ['xtl', 'xtensor', 'xtensor-python', 'evalhyd-cpp']
deps_blank_path = os.path.join(os.getcwd(), 'deps', '{}', 'include')

deps_include_dirs = []
for dep in deps:
    if not os.getenv(f"EVALHYD_PYTHON_VENDOR_{dep.upper().replace('-', '_')}") == 'FALSE':
        # register dependency headers
        deps_include_dirs.append(deps_blank_path.format(dep))
        print(f"vendoring {dep}")

# configure Python extension
ext_modules = [
    Pybind11Extension(
        "evalhyd._evalhyd",
        ['evalhyd/src/evalhyd.cpp'],
        include_dirs=[
            numpy.get_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include'),
            *deps_include_dirs
        ],
        language='c++',
        cxx_std=17,
        define_macros=[('VERSION_INFO', __version__)]
    ),
]

# build Python extension and install Python package
setup(
    name='evalhyd-python',
    version=__version__,
    author='Thibault Hallouin',
    author_email='thibault.hallouin@inrae.fr',
    download_url="https://pypi.python.org/pypi/evalhyd-python",
    project_urls={
        'Bug Tracker': 'https://gitlab.irstea.fr/HYCAR-Hydro/evalhyd/evalhyd-python/-/issues',
        'Documentation': 'https://hydrogr.github.io/evalhyd/python',
        'Source Code': 'https://gitlab.irstea.fr/hycar-hydro/evalhyd/evalhyd-python',
    },
    description='Python bindings for EvalHyd',
    long_description='An evaluator for streamflow predictions.',
    license="GPLv3",
    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ],
    packages=["evalhyd"],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    extras_require={'tests': 'numpy>=1.16'},
    zip_safe=False,
)

import sys
import os
import subprocess
import stat
import shutil

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import numpy


__version__ = '0.0.1'


# vendor dependencies (unless told otherwise via environment variable)
def git_clone(*args):
    r = subprocess.run(
        ['git', 'clone', *args, '-c', 'advice.detachedHead=false'],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8'))


def remove_readonly(func, path, exc):
    # fix for Windows: change access permission for read-only files
    # so that they can be removed
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)


deps = [
    ('xtl', '0.7.0',
     'https://github.com/xtensor-stack/xtl'),
    ('xtensor', '0.24.0',
     'https://github.com/xtensor-stack/xtensor'),
    ('xtensor-python', '0.26.1',
     'https://github.com/xtensor-stack/xtensor-python'),
    ('evalhyd', 'dev',
     'https://gitlab.irstea.fr/HYCAR-Hydro/evalhyd/evalhyd')
]

deps_include_dirs = []
for dep, version, url in deps:
    if not os.getenv(f"EVALHYD_PYTHON_VENDOR_{dep.upper().replace('-', '_')}") == 'FALSE':
        # remove existing dependency if it exists
        dir_path = os.path.join(os.getcwd(), 'deps', dep)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path, onerror=remove_readonly)
            print(f"removed existing {dep}")

        # fetch dependency
        git_clone(url, f'deps/{dep}', '--branch', version)
        print(f"vendored {dep} {version}")

        # register dependency headers
        deps_include_dirs.append(os.path.join(dir_path, 'include'))


# configure Python extension
ext_modules = [
    Pybind11Extension(
        "evalhyd",
        ['src/evalhyd-python.cpp'],
        include_dirs=[
            numpy.get_include(),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include'),
            *deps_include_dirs
        ],
        language='c++',
        define_macros=[('VERSION_INFO', __version__)]
    ),
]

# build Python extension and install Python package
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
    extras_require={"tests": "numpy>=1.16"},
    zip_safe=False,
)

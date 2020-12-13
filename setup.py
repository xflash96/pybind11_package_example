import os
import sys
DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DIR, "third_party", "pybind11"))
print(sys.path)

from glob import glob
from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext # noqa: E402

del sys.path[-1]

pkg_name = 'pybind11_package_example'
ext_name = '_cpp'
__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(pkg_name+'.'+ext_name,
        include_dirs = ['./src'],
        sources = sorted(glob('src/*.c*')),
        define_macros = [('EXTENSION_NAME', ext_name)],
		extra_compile_args = ['-O3', '-Wall', '-g'],
        ),
]

setup(
    name=pkg_name,
    version=__version__,
    install_requires=['numpy'],
    author="Po-Wei Wang",
    author_email="xflash96@gmail.com",
    url="https://github.com/xflash96/pybind11_package_example",
    description="Example python package with pybind11 cpp extension",
    scripts = ['bin/example_cmd'],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    packages=[pkg_name],
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)

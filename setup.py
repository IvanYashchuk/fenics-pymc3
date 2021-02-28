import sys

from setuptools import setup

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = "1.0.0"

setup(
    name="fenics_pymc3",
    description="FEniCS + PyMC3",
    version=version,
    author="Ivan Yashchuk",
    license="MIT",
    packages=["fenics_pymc3"],
    install_requires=["pymc", "fdm", "fecr"],
)

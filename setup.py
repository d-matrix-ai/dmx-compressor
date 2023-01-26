import os
from setuptools import find_packages, setup

VERSION = "0.1.1"

setup(
    name="mltools",
    version=VERSION,
    description="d-Matrix ML tools",
    author="Xin Wang, Michael Klachko, Tristan Webb, Sayeh Sharify, Wanzin Yazar",
    author_email="xwang@d-matrix.ai, mklachko@d-matrix.ai, twebb@d-matrix.ai, sayehs@d-matrix.ai, wyazar@d-matrix.ai",
    license="MIT",
    packages=find_packages("src", exclude=("tests", "docs", "experiments", "sandbox")),
    package_dir={"": "src"},
    package_data={"configs": ["*.yaml"]},
    python_requires=">=3.8",
)

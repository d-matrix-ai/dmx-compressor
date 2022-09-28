import os
from setuptools import find_packages, setup

VERSION = "0.0.3.dev"

setup(
    name="mltools",
    version=VERSION,
    description="d-MATRiX ML tools",
    author="Xin Wang",
    author_email="xwang@d-matrix.ai",
    license="MIT",
    packages=find_packages("src", exclude=("tests", "docs", "experiments", "sandbox")),
    package_dir={"": "src"},
    package_data={
        'configs': ['*.yaml']
    },
    python_requires=">=3.8",
)

from setuptools import find_packages, setup

VERSION = "0.1.2"

setup(
    name="dmx-compressor",
    version=VERSION,
    description="d-Matrix Compressor",
    author="Xin Wang, Michael Klachko, Tristan Webb, Sayeh Sharify, Wanzin Yazar, Zifei Xu",
    author_email="xwang@d-matrix.ai, mklachko@d-matrix.ai, twebb@d-matrix.ai, sayehs@d-matrix.ai, wyazar@d-matrix.ai, xuzifei@d-matrix.ai",
    license="Apache 2.0",
    # packages=find_packages("src", exclude=("tests", "docs", "experiments", "sandbox")),
    packages=["compressor"],
    package_dir={"": "src"},
    package_data={"configs": ["*.yaml"]},
    python_requires=">=3.8",
)

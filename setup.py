import os
from setuptools import find_packages, setup

VERSION = "0.0.3.dev"

DMIR_PROTO_DIR = os.path.join(os.path.dirname(__file__), "src/mltools/utils/")
DMIR_PROTO_FILE = "dmir.proto"

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
    install_requires=(
        "torch>=1.10"
        "torchvision",
        "transformers @ git+ssh://git@git.d-matrix.ai/ml-team/transformers.git@dm-refactor",
        "sol",
        "datasets",
        "numpy",
        "scipy",
        "sklearn",
        "qtorch @ git+ssh://git@git.d-matrix.ai/ml-team/qtorch.git@dm",
        "tensorboard",
        "pyyaml",
        "tqdm",
        "parse",
        "ninja",
        "python-dotenv",
        "pytest",
    ),
    python_requires=">=3.8",
)

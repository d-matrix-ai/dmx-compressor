import os
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

VERSION = "0.0.3.dev"

DMIR_PROTO_DIR = os.path.join(os.path.dirname(__file__), "src/mltools/utils/")
DMIR_PROTO_FILE = "dmir.proto"

class DevelopWrapper(develop):
    """Pre-installation for development mode."""

    def run(self):
        if os.path.exists(os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)):
            os.system(
                f"protoc -I={DMIR_PROTO_DIR} --python_out={DMIR_PROTO_DIR} {os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)}"
            )
        develop.run(self)


class InstallWrapper(install):
    """Pre-installation for installation mode."""

    def run(self):
        if os.path.exists(os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)):
            os.system(
                f"protoc -I={DMIR_PROTO_DIR} --python_out={DMIR_PROTO_DIR} {os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)}"
            )
        install.run(self)

setup(
    name="mltools",
    version=VERSION,
    description="d-MATRiX ML tools",
    author="Xin Wang",
    author_email="xwang@d-matrix.ai",
    license="MIT",
    packages=find_packages("src", exclude=("tests", "docs", "experiments", "sandbox")),
    package_dir={"": "src"},
    install_requires=(
        "torch>=1.9",
        "torchvision",
        "transformers @ git+ssh://git@git.d-matrix.ai/ml-team/transformers.git@dm-refactor",
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
    python_requires=">=3.6",
    cmdclass={
        "develop": DevelopWrapper,
        "install": InstallWrapper,
    },
)

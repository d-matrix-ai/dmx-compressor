import os
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call


DMIR_PROTO_DIR = "./src/utils/"
DMIR_PROTO_FILE = "dmir.proto"

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        if os.path.exists(os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)):
            os.system(f"protoc -I={DMIR_PROTO_DIR} --python_out={DMIR_PROTO_DIR} {os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)}")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        if os.path.exists(os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)):
            os.system(f"protoc -I={DMIR_PROTO_DIR} --python_out={DMIR_PROTO_DIR} {os.path.join(DMIR_PROTO_DIR, DMIR_PROTO_FILE)}")

setup(
    name="compression",
    version="0.0.2.dev",
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
        "qtorch>=0.3",
        "pyyaml",
        "tqdm",
        "parse",
        "ninja",
        "python-dotenv",
        "pytest",
    ),
    python_requires=">=3.6",
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)

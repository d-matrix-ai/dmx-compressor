from setuptools import find_packages, setup

setup(
    name="compression",
    version="0.0.1.dev",
    description="d-MATRiX neural net compression",
    author="Xin Wang",
    author_email="xwang@d-matrix.ai",
    license="MIT",
    packages=find_packages('src', exclude=('tests', 'docs', 'experiments', 'sandbox')),
    package_dir={'': 'src'},
    install_requires=(
        'torch>=1.5',
        'torchvision',
        'transformers',
        'numpy',
        'qtorch>=0.3',
        'tqdm',
        'python-dotenv',
    ),
    python_requires='>=3.6')

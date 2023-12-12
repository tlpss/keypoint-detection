from setuptools import find_packages, setup

setup(
    name="keypoint_detection",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    version="1.0",
    description="Pytorch Models, Modules etc for keypoint detection",
    url="https://github.com/tlpss/keypoint-detection",
    packages=find_packages(exclude=("test",)),
    install_requires=[
        "torch>=0.10",
        "torchvision>=0.11",
        "pytorch-lightning>=1.5.10,<=1.9.4",  # PL 2.0 has breaking changes that need to be incorporated
        "torchmetrics>=0.7",
        "wandb>=0.13.7",  # artifact bug https://github.com/wandb/wandb/issues/4500
        "timm>=0.9",  # 0.9 has breaking changes
        "tqdm",
        "pytest",
        "pre-commit",
        "scikit-image",
        "albumentations",
        "matplotlib",
        "pydantic>=2.0.0",  # 2.0 has breaking changes
        "fiftyone",
    ],
    entry_points={"console_scripts": ["keypoint-detection = keypoint_detection.tasks.cli:main"]},
)

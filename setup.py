from setuptools import setup

setup(
    name="keypoint_detection",
    author="Thomas Lips",
    author_email="thomas.lips@ugent.be",
    version="1.0",
    description="Pytorch Models, Modules etc for keypoint detection",
    url="https://github.com/tlpss/keypoint-detection",
    packages=["keypoint_detection", "labeling"],
    install_requires=[
        "torch>=0.10",
        "torchvision>=0.11",
        "pytorch-lightning>=1.5.10",
        "torchmetrics>=0.7",
        "wandb>=0.13.7",  # artifact bug https://github.com/wandb/wandb/issues/4500
        "timm>=0.6.11",  # requires smallsized convnext models
        "tqdm",
        "pytest",
        "pre-commit",
        "scikit-image",
        "albumentations",
        "matplotlib",
        # for labeling package, should be moved in time to separate setup.py
        "xmltodict",
        "pydantic",
    ],
)

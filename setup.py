from setuptools import setup, find_packages

setup(
    name="effdl",
    author="Tévchhorpoan Khieu",
    description="Tests on the CIFAR-10 dataset",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tevkhieu/cifar-10",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "wandb>=0.12.0",
        "tqdm"
    ],
    python_requires=">=3.7",  # Adjust this depending on your needs
)

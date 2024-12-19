# Setup script for the package


from setuptools import setup, find_packages

setup(
    name="latent_reasoning_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib"
    ]
)

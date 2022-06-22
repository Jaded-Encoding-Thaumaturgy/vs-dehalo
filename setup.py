#!/usr/bin/env python3

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

name = "vsdehalo"
version = "0.1.0"
release = "0.1.0"

setuptools.setup(
    name=name,
    version=release,
    author="Setsugen no ao",
    author_email="setsugen@setsugen.dev",
    description="Collection of dehaloing VapourSynth functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["vsdehalo"],
    url="https://github.com/Irrational-Encoding-Wizardry/vs-dehalo",
    package_data={
        'vsdehalo': ['py.typed'],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10'
)

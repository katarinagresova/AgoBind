from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agobind",
    version="0.1.0",
    author="Katarina Gresova",
    author_email="gresova11@gmail.com",
    description="A package for genomic sequence classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/katarinagresova/agobind",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "torch>=1.11.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "biopython>=1.79",
    ],
) 
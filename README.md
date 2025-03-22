# AgoBind

A Python package for genomic sequence classification tasks.

## Overview

This package facilitates the use of pre-trained models from Hugging Face for embedding and classifying genomic sequences. It provides a streamlined interface for loading genomic datasets, extracting embeddings, fine-tuning models, and evaluating classification performance.

## Installation

### Using conda (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/agobind.git
cd agobind

# Create and activate conda environment
conda env create -f environment.yml
conda activate genomic-ml

# Install package in development mode
pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/agobind.git
cd agobind

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Features

- Load and preprocess genomic sequence data from CSV files
- Access pre-trained genomic models from Hugging Face
- Extract embeddings from genomic sequences
- Fine-tune models for classification tasks
- Evaluate model performance with specialized metrics

## Usage

Example code will be provided once the initial implementation is complete.

## Development

See [PLAN.md](PLAN.md) for development roadmap and implementation details.

## License

MIT License 
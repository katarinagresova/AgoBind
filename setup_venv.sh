#!/bin/bash
# Setup script for AgoBind environment using venv

# Set strict error handling
set -e

# Echo commands
set -x

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python 3 first."
    exit 1
fi

# Create a virtual environment
echo "Creating Python virtual environment..."
python3 -m venv agobind_venv

# Activate the environment
echo "Activating environment..."
source agobind_venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install the required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install additional packages for genomic models
echo "Installing additional packages..."
pip install einops accelerate sentencepiece wandb plotly umap-learn tensorboardX

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create a kernel for Jupyter (if jupyter is installed)
if python -c "import jupyter" &> /dev/null; then
    echo "Setting up Jupyter kernel..."
    python -m ipykernel install --user --name agobind_venv --display-name "Python (AgoBind venv)"
    echo "Jupyter kernel installed. Select 'Python (AgoBind venv)' in Jupyter."
else
    echo "Jupyter not found. If you want to use Jupyter, install it with: pip install jupyter ipykernel"
fi

echo "Environment setup complete!"
echo "To activate the environment, run: source agobind_venv/bin/activate" 
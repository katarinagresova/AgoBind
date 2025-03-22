#!/bin/bash
# Setup script for AgoBind environment

# Set strict error handling
set -e

# Echo commands
set -x

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "agobind-env"; then
    read -p "Environment 'agobind-env' already exists. Do you want to remove it and create a new one? (y/n) " choice
    if [[ "$choice" == "y" ]]; then
        conda env remove -n agobind-env
    else
        echo "Using existing environment. Some packages may need to be updated manually."
    fi
fi

# Create the conda environment
echo "Creating conda environment from environment_complete.yml..."
conda env create -f environment_complete.yml

# Activate the environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate agobind-env

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create a kernel for Jupyter
python -m ipykernel install --user --name agobind-env --display-name "Python (AgoBind)"

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate agobind-env"
echo "To use in Jupyter, select the 'Python (AgoBind)' kernel" 
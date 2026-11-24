#!/bin/bash
# ==============================================================================
# Helper script to set up the environment on Demetra HPC (UniTS)
# ==============================================================================
set -e

echo "=== Demetra HPC Environment Setup for LTLfBERT ==="
echo "Select installation mode:"
echo "  1) Conda / Miniforge (RECOMMENDED: precompiled Spot binary via conda-forge)"
echo "  2) Python venv (Compiles Spot via pip into a virtualenv)"
read -p "Enter choice [1 or 2]: " CHOICE

if [ "$CHOICE" == "1" ]; then
    echo "--- Installing via Conda ---"
    if ! command -v conda &> /dev/null; then
        echo "Conda not found in PATH. Checking environment modules..."
        module avail miniforge 2>/dev/null || module avail anaconda 2>/dev/null || true
        echo "If available, run 'module load miniforge' or install Miniforge locally in your home:"
        echo "  curl -L -O \"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh\""
        echo "  bash Miniforge3-\$(uname)-\$(uname -m).sh"
        exit 1
    fi
    echo "Creating conda environment 'ltlfbert' with Spot, DEAP, and NumPy..."
    conda create -y -n ltlfbert -c conda-forge python=3.11 spot deap numpy scipy
    echo "Environment created! Activate it using: conda activate ltlfbert"

elif [ "$CHOICE" == "2" ]; then
    echo "--- Installing via Python venv ---"
    echo "Checking available Python modules..."
    module avail python 2>/dev/null || true
    echo "Tip: load gcc and a modern python module first if needed, e.g.:"
    echo "  module load gcc python/3.11"
    
    mkdir -p "$HOME/envs"
    python3 -m venv "$HOME/envs/ltlfbert"
    source "$HOME/envs/ltlfbert/bin/activate"
    
    pip install --upgrade pip setuptools wheel
    pip install numpy scipy deap
    
    echo "Installing Spot (compiling from official LRDE tarball)..."
    pip install http://www.lre.epita.fr/dload/spot/spot-2.16.1.tar.gz
    
    echo "Virtual environment created at: $HOME/envs/ltlfbert"
    echo "Activate it using: source ~/envs/ltlfbert/bin/activate"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo "=== Setup complete! Verify with: python3 -c \"import spot, deap; print('OK')\" ==="

#!/bin/bash
# ============================================================================
# Physics-SR Framework v3.0 - Google Colab Setup Script
# ============================================================================
# Run this script in Colab to set up the environment:
#   !bash setup_colab.sh
# ============================================================================

echo "=============================================="
echo "Physics-SR Framework v3.0 - Colab Setup"
echo "=============================================="

# Colab already has: numpy, pandas, scipy, sklearn, sympy, matplotlib, seaborn, tqdm
# We only need to install PySR

echo "[1/2] Installing PySR (the only package not in Colab)..."
pip install -q pysr

# Initialize PySR (installs Julia backend, takes 2-3 minutes first time)
echo "[2/2] Initializing PySR Julia backend (this takes 2-3 minutes)..."
python -c "
import pysr
try:
    pysr.install()
    print('PySR Julia backend installed successfully.')
except Exception as e:
    print(f'PySR status: {e}')
"

echo "=============================================="
echo "Setup complete! Ready to run experiments."
echo "=============================================="

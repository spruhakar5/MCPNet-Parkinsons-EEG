#!/bin/bash
# MCPNet Setup Script
# Downloads dependencies and EEG datasets automatically

set -e

echo "============================================"
echo "  MCPNet Setup: EEG Parkinson's Detection"
echo "============================================"

# Install Python dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt
pip install openneuro-py awscli 2>/dev/null || true

# Create data directories
echo ""
echo "[2/3] Creating data directories..."
mkdir -p data/raw/UC data/raw/UNM data/raw/Iowa data/processed

# Download datasets
echo ""
echo "[3/3] Downloading EEG datasets from OpenNeuro..."
echo "  This may take a while depending on your connection."
echo ""
cd src && python download_data.py --dataset all
cd ..

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To run the pipeline:"
echo "    cd src"
echo "    python main.py --real --k_shot 5"
echo ""
echo "  Or test with synthetic data first:"
echo "    python main.py --n_subjects 10 --k_shot 5"
echo "============================================"

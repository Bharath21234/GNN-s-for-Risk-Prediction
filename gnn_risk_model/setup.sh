#!/usr/bin/env bash
# ── GNN Risk Model — Environment Setup ─────────────────────────────────────────
# Run once from inside the gnn_risk_model/ directory.
# Requires conda (Miniconda or Anaconda) to be installed.
set -e

echo "==================================================================="
echo "  GNN-Based Equity Tail Risk Model — Environment Setup"
echo "==================================================================="

# ── 1. Create conda environment ────────────────────────────────────────────────
echo ""
echo "[1/4] Creating conda environment 'gnn-risk' (Python 3.11)..."
conda env create -f environment.yml --force
echo "  Done."

# ── 2. Activate ────────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnn-risk

# ── 3. Install PyG extensions (optional, speeds up message passing) ────────────
echo ""
echo "[3/4] Installing torch-geometric extras (optional)..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "  Detected torch version: $TORCH_VERSION"
pip install torch-scatter torch-sparse -f \
    "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html" \
    --quiet || echo "  (Extras skipped — basic torch-geometric will still work)"

# ── 4. Create directories ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Creating output directories..."
mkdir -p data/raw data/processed models results/plots

echo ""
echo "==================================================================="
echo "  Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    conda activate gnn-risk"
echo ""
echo "  Then run the pipeline in order:"
echo "    python 1_download_data.py   # ~2-5 min  — fetch prices from Yahoo"
echo "    python 2_build_features.py  # ~3-8 min  — features + graph"
echo "    python 3_train_gnn.py       # ~10-30 min — train GraphSAGE"
echo "    python 4_garch_baselines.py # ~10-25 min — GARCH + GJR-GARCH"
echo "    python 5_evaluate.py        # ~1 min    — compare + plot"
echo ""
echo "  Results land in results/"
echo "==================================================================="

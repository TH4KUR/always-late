#!/bin/bash
# Setup script for Gemma-2-9B confirmation runs on a fresh VM.
# Prerequisites: CUDA-capable GPU (>=40 GB VRAM recommended; 2x smaller GPUs possible with n_devices=2),
#                Python 3.10+, git, and a Hugging Face token with Gemma-2 access.
#
# Usage:
#   export HF_TOKEN=hf_...        # your HF token (must have accepted Gemma-2 terms)
#   bash scripts/g0/setup_gemma9b_vm.sh [BASE_DIR]
#
# BASE_DIR defaults to ~/gemma9b-workspace

set -euo pipefail

BASE=${1:-$HOME/gemma9b-workspace}
mkdir -p "$BASE"
cd "$BASE"

echo "=== Setting up Gemma-2-9B workspace in $BASE ==="

# ── 1. Python virtual environment ──────────────────────────────────────────────
if [ -d "env/bin" ]; then
    echo "[1/5] Python virtual environment already exists, reusing it."
else
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv env
fi
source env/bin/activate

# ── 2. Install pinned packages (versions matching the 2B study on Ada) ─────────
if python -c "import torch, transformer_lens, sae_lens" 2>/dev/null; then
    echo "[2/5] Packages already installed, skipping."
else
    echo "[2/5] Installing Python packages..."
    pip install --upgrade pip

    pip install \
    "torch==2.14.1" \
    "transformer-lens==3.9.0" \
    "sae-lens==6.54.5" \
    "transformers==5.19.0" \
    "safetensors==0.8.0" \
    "huggingface_hub==2.2.0" \
    "wandb==0.30.0" \
    "numpy==2.2.6" \
    "scipy==1.15.3" \
    "tqdm==4.70.1" \
    "einops==0.8.2" \
    "jaxtyping==0.3.7" \
    "datasets==5.1.0" \
    "sentencepiece==0.2.2" \
    "tokenizers==0.23.3" \
    "pandas==2.3.3" \
    "matplotlib==3.10.9" \
    "scikit-learn==1.7.2" \
    "statsmodels==0.15.0" \
    "ipython==8.39.0"
fi

echo "Installed versions:"
python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from importlib.metadata import version; print(f'  transformer-lens {version(\"transformer-lens\")}')"
python -c "import sae_lens; print(f'  sae-lens {sae_lens.__version__}')"
python -c "import transformers; print(f'  transformers {transformers.__version__}')"

# ── 3. Download models and SAEs from Hugging Face ──────────────────────────────
echo "[3/5] Downloading Gemma-2-9B model and Gemma Scope SAE..."

echo "Make sure you are logged in to Hugging Face: run 'hf auth login' first (inside the venv)."
echo "You must have accepted the Gemma-2 terms at https://huggingface.co/google/gemma-2-9b"

# Gemma-2-9B model (~18 GB)
echo "  Downloading google/gemma-2-9b..."
hf download google/gemma-2-9b

# Gemma Scope 9B SAE (only the layer we need)
echo "  Downloading google/gemma-scope-9b-pt-res (layer_20/width_16k/canonical)..."
hf download google/gemma-scope-9b-pt-res \
    --include "layer_20/width_16k/canonical/*"

# Also grab the 2B model and SAE (needed for the 2B vs 9B comparison, and the frozen runners reference it)
echo "  Downloading google/gemma-2-2b..."
hf download google/gemma-2-2b

echo "  Downloading google/gemma-scope-2b-pt-res (layer_12/width_16k/canonical)..."
hf download google/gemma-scope-2b-pt-res \
    --include "layer_12/width_16k/canonical/*"

# ── 4. Download FLORES-200 dataset ────────────────────────────────────────────
echo "[4/5] Downloading FLORES-200 dataset..."
FLORES_DIR="$BASE/data"
mkdir -p "$FLORES_DIR"
cd "$FLORES_DIR"

if [ ! -d "flores200_dataset" ]; then
    echo "  Fetching the ORIGINAL FLORES-200 archive (not FLORES+)..."
    wget -q --show-progress "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    tar xzf flores200_dataset.tar.gz
    rm flores200_dataset.tar.gz
    echo "  Verifying structure..."
    ls flores200_dataset/dev/eng_Latn.dev > /dev/null
    ls flores200_dataset/devtest/eng_Latn.devtest > /dev/null
    DEV_COUNT=$(wc -l < flores200_dataset/dev/eng_Latn.dev)
    DEVTEST_COUNT=$(wc -l < flores200_dataset/devtest/eng_Latn.devtest)
    echo "  eng_Latn: $DEV_COUNT dev lines, $DEVTEST_COUNT devtest lines (expect 997 / 1012)"
else
    echo "  flores200_dataset already exists, skipping download."
fi
cd "$BASE"

# ── 5. Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Setup complete."
echo ""
echo "Workspace:    $BASE"
echo "Python env:   $BASE/env  (activate with: source $BASE/env/bin/activate)"
echo "HF cache:     ~/.cache/huggingface (default)"
echo "FLORES data:  $BASE/data/flores200_dataset"
echo ""
echo "Next steps:"
echo "  1. Clone the repo:     git clone https://github.com/TH4KUR/always-late.git && cd always-late && git checkout gemma9b-confirmation"
echo "  2. Run the tests:      source $BASE/env/bin/activate && python3 -B -m unittest discover -s scripts/g0/tests"
echo "  3. Run torch tests:    OMP_NUM_THREADS=4 python -B -m unittest discover -s scripts/g0/tests_torch"
echo "  4. Build the adapter, write the freeze, then run the experiments per the plan in docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md"
echo ""
echo "Key paths for the scripts:"
echo "  FLORES_ROOT=$BASE/data"

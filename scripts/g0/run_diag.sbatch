#!/bin/bash
# Run one diagnostic script from the deployed Phase 1 repo on ONE 2080 Ti, offline from the persistent HF cache. Usage: sbatch [-A nlp] run_diag.sbatch diag_loss.py
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -J diag
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE/phase1/repo/scripts/g0
$BASE/env/bin/python "$@"

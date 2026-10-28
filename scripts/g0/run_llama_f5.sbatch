#!/bin/bash
# F5 (single-feature sameness of function) on Llama-3.2-1B with the FineWeb TopK SAE (Stage 5 of docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md; addendum
# reports/survival_effect_rescue/llama/llama_f5_addendum_2026-10-28.md): the frozen F5 runner through scripts/g0/phase1_llama_run.py (task f5) on the Llama sample reports/survival_effect_rescue/llama/llama_f5_sample.json.
# Usage: sbatch [-t HH:MM:SS] --exclude=gnode068,gnode069 scripts/g0/run_llama_f5.sbatch <name> [args of phase1_func_f5_run.py without --out, --flores-root, --sample, --commit]
#   smoke  : sbatch -t 01:00:00 run_llama_f5.sbatch llama-f5-smoke --states FP 16 4 --max-features 24 --max-eval 24 --dev-sentences 60 --per-cell 12 --subset-seed 0
#   full   : sbatch -t 04:00:00 run_llama_f5.sbatch llama-f5 --states FP 16 6 4 --per-cell 12 --subset-seed 0
# Read the `GPU 0:` line AND the first lines of the log for `CUDA initialization` (a node whose CUDA fails runs on the CPU and stops at the first gate). The Llama weights and SAEs are in the cache of HF_HOME below;
# the licence of the weights applies: nothing derived from them that reproduces them leaves this machine.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J llama-f5
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; shift
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$BASE/logs"
ls $HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/*/model.safetensors > /dev/null || { echo "Llama weights missing in the cache"; exit 3; }
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)      # REPO may point to a snapshot deployed under another name (deploy_phase1.sh DEST)
nvidia-smi -L; date
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
PY=$BASE/env/bin/python
$PY phase1_llama_run.py f5 --out "$OUT" --flores-root $BASE/phase1/data --sample $REPO/reports/survival_effect_rescue/llama/llama_f5_sample.json --commit "$COMMIT" "$@"
date

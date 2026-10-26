#!/bin/bash
# Intervention-transfer study (D-088), the feature-level core on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB), Gemma-2-2B layer 12: scripts/g0/phase1_transfer_run.py.
# Usage: sbatch [-t HH:MM:SS] --exclude=gnode068,gnode069 --export=ALL,REPO=<snapshot> scripts/g0/run_transfer.sbatch <name> [extra args of phase1_transfer_run.py]
# instrument job (plan Task 2, a correctness and throughput gate, NOT a scientific pilot): sbatch -t 00:40:00 --exclude=gnode068,gnode069 ... run_transfer.sbatch transfer-instrument --states 16 4 --max-features 12 --max-contexts 4 --chunk-features 6
# full run: one job per quantized state keeps each job near three hours:  run_transfer.sbatch transfer-int6 --states 6   (and 4, 16)
# Prereqs: code deployed by scripts/g0/deploy_phase1.sh (it ships reports/intervention_transfer/freeze.json and core_sample.json); env, model cache and FLORES as for run_f5.sbatch.
# Read the `GPU 0:` line AND the first lines of the log for `CUDA initialization` (a node whose CUDA fails runs on the CPU and stops at a gate). Jobs run offline (HF_HUB_OFFLINE=1).
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J transfer
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; shift
BASE=/home2/karma/duan-verify
export HF_TOKEN_PATH=$BASE/hf/token
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$BASE/logs"
ls $HF_HOME/hub/models--google--gemma-2-2b/snapshots/*/model-00003-of-00003.safetensors > /dev/null || { echo "model cache missing: run fetch_model.sh on the login node"; exit 3; }
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)      # REPO may point to a snapshot deployed under another name (deploy_phase1.sh DEST)
nvidia-smi -L; date
nvidia-smi -L | grep -q 'GPU 0' || { echo 'no GPU visible on this node: refusing to run on the CPU (job 40402 did, 2026-10-26)'; exit 4; }     # a node whose CUDA fails would run on the CPU and still pass every gate
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_transfer_run.py --out "$OUT" --flores-root $BASE/phase1/data --freeze $REPO/reports/intervention_transfer/freeze.json \
    --core $REPO/reports/intervention_transfer/core_sample.json --commit "$COMMIT" "$@"
date

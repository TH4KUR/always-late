#!/bin/bash
# Gemma-2-9B confirmation (D-089, handoff 040): the frozen Gemma runners through scripts/g0/phase1_gemma9b_run.py on ONE GPU with >= 40 GB (A100, A6000, L40S, H100).
# The 9B weights are 18.5 GB in bf16 and do not fit one 11 GB RTX 2080 Ti: there, split the model over 4 cards (export GEMMA9B_N_DEVICES=4 and submit with --gres=gpu:4 --constraint=2080ti);
# on a card of >= 40 GB leave GEMMA9B_N_DEVICES unset (1). Pass the partition, GPU type and count on the command line.
# Usage: sbatch [-t HH:MM:SS] [--constraint=..] scripts/g0/run_gemma9b.sbatch <name> tag [extra args of phase1_gemma9b_run.py tag]
#        sbatch [-t HH:MM:SS] scripts/g0/run_gemma9b.sbatch <name> manifest <tag job name> [--families full|k]          (CPU work; rebuilds the manifest, e.g. with --families k)
#        sbatch [-t HH:MM:SS] scripts/g0/run_gemma9b.sbatch <name> lm <tag job name> [args of phase1_func_run.py without --preset, --ratios, --selectivity, --manifest]
# Run it from the repository root (the log goes to ./logs/, create it first: mkdir -p logs).
# Environment (export before sbatch; sbatch passes it on):
#   BASE      work directory holding hf_cache/, data/ (with flores200_dataset/) and out/phase1/   (default $HOME/gemma9b)
#   REPO      checkout of this repo                                                              (default: the repository of this script; under sbatch pass REPO explicitly, the script is copied)
#   PY        python of the environment with torch, transformer-lens 3.9.0, sae-lens, transformers (default python)
#   GEMMA9B_N_DEVICES  GPUs the model is split over, 1 (default) or 4 (Ada 2080 Ti); must match --gres=gpu:N
#   HF_TOKEN  Hugging Face token with the Gemma licence accepted (needed on the first, online, download)
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 30:00:00
#SBATCH -J gemma9b
#SBATCH -o logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; TASK=${2:?tag, manifest or lm}; shift 2
BASE=${BASE:-$HOME/gemma9b}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
PY=${PY:-python}
export HF_HOME=${HF_HOME:-$BASE/hf_cache}
export GEMMA9B_N_DEVICES=${GEMMA9B_N_DEVICES:-1}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE/out/phase1" logs
COMMIT=$(cat "$REPO/COMMIT" 2>/dev/null || git -C "$REPO" rev-parse HEAD)
OUT=$BASE/out/phase1/$NAME; mkdir -p "$OUT"
nvidia-smi -L || true; date; echo "commit $COMMIT repo $REPO base $BASE"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" 2>/dev/null &
MONPID=$!
trap 'kill ${MONPID:-0} 2>/dev/null || true' EXIT
cd "$REPO/scripts/g0"
if [ "$TASK" = "tag" ]; then
    $PY phase1_gemma9b_run.py tag --out "$OUT" --flores-root "$BASE/data" --commit "$COMMIT" "$@"
    $PY phase1_gemma9b_run.py manifest --selectivity "$OUT/eq/selectivity.pt" --manifest "$OUT/groups_manifest.json"
elif [ "$TASK" = "manifest" ]; then
    TAG=${1:?tag job name}; shift
    $PY phase1_gemma9b_run.py manifest --selectivity "$BASE/out/phase1/$TAG/eq/selectivity.pt" --manifest "$BASE/out/phase1/$TAG/groups_manifest.json" "$@"
elif [ "$TASK" = "lm" ]; then
    TAG=${1:?tag job name}; shift
    $PY phase1_gemma9b_run.py lm --out "$OUT" --flores-root "$BASE/data" --selectivity "$BASE/out/phase1/$TAG/eq/selectivity.pt" --manifest "$BASE/out/phase1/$TAG/groups_manifest.json" --commit "$COMMIT" "$@"
else
    echo "unknown task $TASK"; exit 2
fi
date

#!/bin/bash
# Llama-3.2-1B consistency check (plan docs/superpowers/plans/2026-10-26-llama-consistency-check.md, D-087) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB): the frozen Gemma runners through scripts/g0/phase1_llama_run.py.
# Usage: sbatch [-t HH:MM:SS] --exclude=gnode068,gnode069 scripts/g0/run_llama.sbatch <name> tag [--sae fineweb42|faithful42] [extra args of phase1_llama_run.py tag]
#        sbatch [-t HH:MM:SS] --exclude=gnode068,gnode069 scripts/g0/run_llama.sbatch <name> lm <tag job name> [--sae ...] [args of phase1_func_run.py without --preset, --ratios, --selectivity, --manifest]
# tag : the dev tagging pass (all 27 languages, equal token budget) -> $OUT/eq/selectivity.pt and meta.json, then the frozen group builder with the K16 and K64 families -> $OUT/groups_manifest.json (CPU, seconds; the `all` family of the Gemma manifest cannot be built on this SAE: the pool is smaller than the whole group of English).
# lm  : the frozen T-LM runner with the Study G down-ratio preset on the files of the tag job; e.g.
#       smoke : sbatch -t 01:00:00 run_llama.sbatch llama-smoke-lm llama-smoke-tag --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel
#       full  : sbatch -t 04:00:00 run_llama.sbatch llama-lm-a llama-tag --precisions FP 8 6 --split eval --n-eval 300 --panel eng_Latn hin_Deva     (and llama-lm-b: --precisions 5 4)
# Read the `GPU 0:` line AND the first lines of the log for `CUDA initialization` (a node whose CUDA fails runs on the CPU and stops at the first gate). The Llama weights and SAEs are in the cache of the login node
# (HF_HOME below); the licence of the weights applies: nothing derived from them that reproduces them leaves this machine.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -J llama
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?name}; TASK=${2:?tag or lm}; shift 2
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
if [ "$TASK" = "tag" ]; then
    $PY phase1_llama_run.py tag --out "$OUT" --flores-root $BASE/phase1/data --commit "$COMMIT" "$@"
    $PY phase1_llama_run.py manifest --selectivity "$OUT/eq/selectivity.pt" --manifest "$OUT/groups_manifest.json"
elif [ "$TASK" = "lm" ]; then
    TAG=${1:?tag job name}; shift
    $PY phase1_llama_run.py lm --out "$OUT" --flores-root $BASE/phase1/data --selectivity $BASE/out/phase1/$TAG/eq/selectivity.pt --manifest $BASE/out/phase1/$TAG/groups_manifest.json --commit "$COMMIT" "$@"
else
    echo "unknown task $TASK"; exit 2
fi
date

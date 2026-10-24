#!/bin/bash
# Study G (graded gain, D-070) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB): the frozen functional-validation runner wrapped by scripts/g0/phase1_gain_run.py, which registers the preset `gain|<ratios>|c<n>|K64`.
# Usage: sbatch [-t HH:MM:SS] scripts/g0/run_gain.sbatch <name> --ratios R [R ...] [--n-controls 3] [extra args for phase1_func_run.py WITHOUT --preset ...]
# smoke       : sbatch -t 01:00:00 run_gain.sbatch gain-smoke --ratios 0.5 2 --precisions FP 4 --languages eng_Latn hin_Deva khm_Khmr --max-sentences 40 --split pilot --panel
# calibration : sbatch -t 04:00:00 run_gain.sbatch gain-cal-<p> --ratios <18 ratios> --n-controls 2 --precisions FP|4 --split pilot --panel
# main        : sbatch -t 06:00:00 run_gain.sbatch gain-a --ratios <8 ratios> --precisions FP 8 7 --split eval --n-eval 300 --panel     (and gain-b: 6 5 4)
# Prereqs: code deployed by scripts/g0/deploy_phase1.sh (it also ships reports/functional_validation/groups_manifest.json); env, model cache, FLORES and the original p1-mask/eq/selectivity.pt as for run_phase1.sbatch.
# NOTE: /share1 is NOT mounted on compute nodes; everything lives on /home2. Jobs run offline.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 06:00:00
#SBATCH -J gain
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
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 10 > "$OUT/gpu_mem.log" &
MONPID=$!
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_gain_run.py --out "$OUT" --flores-root $BASE/phase1/data --selectivity $BASE/out/phase1/p1-mask/eq/selectivity.pt \
    --manifest $REPO/reports/functional_validation/groups_manifest.json --commit "$COMMIT" "$@"
date

#!/bin/bash
# Task 17 part (b) (prompt cross-check of the Belebele pipeline) on ONE GPU of Ada (u22, RTX 2080 Ti 11 GB): the frozen T-MC runner with the frozen letter prompt (variant frozen) and with the lm-evaluation-harness
# Belebele prompt (variant harness), one process each, one after the other, through scripts/g0/phase1_func_promptcheck.py (plan 2026-10-21-functional-validation.md, last bullet; source S08).
# Usage: sbatch [-t HH:MM:SS] --exclude=gnode068 scripts/g0/run_promptcheck.sbatch <name> --languages L [L ...] [overrides of the fixed arguments below, e.g. --n-items 40]
# smoke : sbatch -t 01:00:00 --exclude=gnode068 run_promptcheck.sbatch promptcheck-smoke --languages eng_Latn hin_Deva --n-items 40
# full  : sbatch -t 02:00:00 --exclude=gnode068 run_promptcheck.sbatch promptcheck --languages <the 18 languages of evaluation_freeze_mc.json calibration.covered>
# Fixed arguments (the calibration run of the frozen MC statement): calibration items, letter format, no edit (preset none), full precision, 250 items per language, item seed 0, the frozen defaults otherwise.
# Output: $OUT/frozen/ and $OUT/harness/ with func_mc_cal_letter_FP.pt and meta_mc_cal.json each. Read the `GPU 0:` line of the log: it must be an RTX 2080 Ti (gnode068 is excluded, D-079).
# Prereqs: code deployed by scripts/g0/deploy_phase1.sh (named snapshot; REPO points to it); env, model cache, FLORES, Belebele files and the original p1-mask/eq/selectivity.pt as for run_functional.sbatch.
# NOTE: /share1 is NOT mounted on compute nodes; everything lives on /home2. Jobs run offline.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -J promptcheck
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
COMMON=(--belebele-root $BASE/phase1/data/belebele --origin-file $BASE/phase1/data/belebele/passage_origin.json --flores-root $BASE/phase1/data
        --selectivity $BASE/out/phase1/p1-mask/eq/selectivity.pt --manifest $REPO/reports/functional_validation/groups_manifest.json --commit "$COMMIT"
        --split cal --formats letter --preset none --precisions FP --n-items 250 --item-seed 0)
for V in frozen harness; do
    echo "=== variant $V"; date
    $BASE/env/bin/python phase1_func_promptcheck.py --variant $V --out "$OUT/$V" "${COMMON[@]}" "$@"
done
date

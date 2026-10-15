#!/bin/bash
# BASELINE REPLICATION of Duan (2026) Phase 3 on Ada: runs THEIR script (faithful runner: only [FIX] runnable export, [PLACE] second-card
# placement, [MODE] their own 3a_test|3b_full switch; see make_faithful_runner.py) and then compares with their published table.
# Usage: sbatch scripts/g0/run_baseline.sbatch <arm-name> <3a_test|3b_full>
#   smoke:    sbatch -J base-smoke scripts/g0/run_baseline.sbatch smoke 3a_test        (~6 min compute + ~18 min model download)
#   baseline: sbatch -J base-full  scripts/g0/run_baseline.sbatch full  3b_full        (their shipped 500k-token run incl. INT4/5 and pruning: ~5.5 h)
# NOTE: /share1 is NOT mounted on compute nodes; everything lives on /home2 or node-local /scratch.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 14:00:00
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
ARM=${1:?arm}; MODE=${2:?3a_test|3b_full}
V=/home2/karma/duan-verify
export HF_TOKEN_PATH=$V/hf/token
export HF_HOME=/scratch/karma-g0-hf-${SLURM_JOB_ID:-manual}      # model (10.5 GB fp32) on node-local scratch, deleted at job end
export HF_XET_CACHE=$HF_HOME/xet HF_XET_HIGH_PERFORMANCE=1       # do NOT set HF_HUB_DISABLE_XET=1 (15x slower, drops connections)
mkdir -p "$HF_HOME" "$V/logs"
trap 'kill ${MONPID:-0} 2>/dev/null || true; rm -rf "$HF_HOME"' EXIT
for attempt in 1 2 3 4 5 6; do
  $V/env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json && break
  echo "model download attempt $attempt failed; retrying"; sleep 10
done
export G0_WORK_DEVICE=cuda:1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export WANDB_PROJECT=${WANDB_PROJECT:-anlp-project} G0_WANDB_NAME=${G0_WANDB_NAME:-baseline-$ARM-$SLURM_JOB_ID}
OUT=$V/baseline/$ARM; mkdir -p "$OUT"; cd "$OUT"                  # their script writes phase3_gemma_outputs_<mode>_<k>k_L12 relative to the cwd
nvidia-smi -L; date
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_mem.log" &
MONPID=$!
$V/env/bin/python $V/faithful_$MODE.py                           # THEIR code
date
RUN=$(ls -d "$OUT"/phase3_gemma_outputs_${MODE}_*_L12 | head -1)
$V/env/bin/python $V/baseline_report.py --run-dir "$RUN" --published $V/published/phase3_summary_final.csv \
   --per-feature "$RUN/RTN_INT6_per_feature.csv" --published-per-feature $V/published/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv --wandb || echo "report step failed (non-fatal)"

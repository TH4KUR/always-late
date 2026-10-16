#!/bin/bash
# BASELINE REPLICATION of Duan (2026) Phase 3 on Ada. Runs THEIR vendored script (third_party/duan2026/scripts/03_gemma_phase3_streaming_sweep.py =
# their file + 5 whitelisted device/flag edits, see third_party/duan2026/UPSTREAM.md) under IPython (their notebook-only display() needs no edit),
# then compares with their published table and uploads to wandb (post hoc, from their own output files).
# Usage: sbatch -J <name> scripts/g0/run_baseline.sbatch <arm> <full|smoke>
#   full : their shipped 3b_full configuration (500k tokens, INT8/7/6/5/4) with their pruning flag off          (~4.1 h compute)
#   smoke: the same file with the single line  RUN_MODE = "3b_full"  replaced by "3a_test" in a TEMPORARY copy   (their shipped smoke mode, ~6 min)
# NOTE: /share1 is NOT mounted on compute nodes; everything lives on /home2 or node-local /scratch.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 14:00:00
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
ARM=${1:?arm}; KIND=${2:?full|smoke}
V=/home2/karma/duan-verify
# --- refuse to run unless the deployed script is byte-identical to the vendored, tested one
( cd $V/vendor && sha256sum -c working.sha256 ) || { echo "ABORT: vendored script hash mismatch"; exit 3; }
export HF_TOKEN_PATH=$V/hf/token
export HF_HOME=/scratch/karma-g0-hf-${SLURM_JOB_ID:-manual}      # gated model (10.5 GB fp32) on node-local scratch, deleted at job end
export HF_XET_CACHE=$HF_HOME/xet HF_XET_HIGH_PERFORMANCE=1       # do NOT set HF_HUB_DISABLE_XET=1 (15x slower, drops connections)
mkdir -p "$HF_HOME" "$V/logs"
trap 'kill ${MONPID:-0} 2>/dev/null || true; rm -rf "$HF_HOME"' EXIT
for attempt in 1 2 3 4 5 6; do
  $V/env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json && break
  echo "model download attempt $attempt failed; retrying"; sleep 10
done
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export WANDB_PROJECT=${WANDB_PROJECT:-anlp-project} G0_WANDB_NAME=${G0_WANDB_NAME:-baseline/$ARM}
OUT=$V/baseline/$ARM; mkdir -p "$OUT"; cd "$OUT"                  # their script writes phase3_gemma_outputs_<mode>_<k>k_L12 relative to the cwd
SRC=$V/vendor/scripts/03_gemma_phase3_streaming_sweep.py
if [ "$KIND" = "smoke" ]; then
  cp "$SRC" "$OUT/run_03_smoke.py"; sed -i 's/^RUN_MODE = "3b_full"$/RUN_MODE = "3a_test"/' "$OUT/run_03_smoke.py"
  echo "smoke copy differs from the vendored file by:"; diff "$SRC" "$OUT/run_03_smoke.py" || true
  MODE=3a_test; RUNFILE="$OUT/run_03_smoke.py"
else
  MODE=3b_full; RUNFILE="$SRC"
fi
nvidia-smi -L; date; echo "running: $RUNFILE  sha256 $(sha256sum "$RUNFILE" | cut -c1-16)"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_mem.log" &
MONPID=$!
$V/env/bin/ipython "$RUNFILE"                                    # THEIR code (under IPython so their notebook-only display() works)
date
RUN=$(ls -d "$OUT"/phase3_gemma_outputs_${MODE}_*_L12 | head -1)
$V/env/bin/python $V/baseline_report.py --run-dir "$RUN" --published $V/published/phase3_summary_final.csv \
   --per-feature "$RUN/RTN_INT6_per_feature.csv" --published-per-feature $V/published/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv --wandb || echo "report step failed (non-fatal)"

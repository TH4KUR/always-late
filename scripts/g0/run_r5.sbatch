#!/bin/bash
# R5 (our pipeline on THEIR tokens, one factor flipped per cell). 1 x RTX 2080 Ti (11 GB): one bf16 model (5.2 GB) quantized in place.
# Usage: sbatch -J R5a scripts/g0/run_r5.sbatch <out-name> <cells> [bits=8,7,6] [extra r5_run.py args]
#   e.g. sbatch -J R5a run_r5.sbatch R5a T,O,T_sdpa,T_bf16sae        smoke: sbatch -J R5smoke run_r5.sbatch R5smoke T,O 8 --limit-windows 40 --selfcheck-windows 4
# Resumable: finished cells (cell_done.json) are skipped when the same <out-name> is submitted again.
# Prereqs as run_g0.sbatch: env /home2/karma/duan-verify/env; HF token at /home2/karma/duan-verify/hf/token; code deployed to /home2/karma/duan-verify/r5code (r5_run.py r5_lib.py g0_wandb.py src/).
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 08:00:00
#SBATCH -J r5
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
NAME=${1:?out-name}; CELLS=${2:?cells}; BITS=${3:-8,7,6}; shift $(( $# < 3 ? $# : 3 ))
export HF_TOKEN_PATH=/home2/karma/duan-verify/hf/token
export HF_HOME=/scratch/karma-r5-hf-${SLURM_JOB_ID:-manual}
export HF_XET_CACHE=$HF_HOME/xet HF_XET_HIGH_PERFORMANCE=1      # NOTE: never set HF_HUB_DISABLE_XET=1 (15x slower on this cluster)
mkdir -p "$HF_HOME" /home2/karma/duan-verify/logs
trap 'kill ${MONPID:-0} 2>/dev/null || true; rm -rf "$HF_HOME"' EXIT
for attempt in 1 2 3 4 5 6; do
  /home2/karma/duan-verify/env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json && break
  echo "model download attempt $attempt failed; retrying"; sleep 10
done
du -sh "$HF_HOME"
OUT=/home2/karma/duan-verify/r5/$NAME; mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_PROJECT=${WANDB_PROJECT:-anlp-project}
nvidia-smi -L; date
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_mem.log" &
MONPID=$!
cd /home2/karma/duan-verify/r5code
sha256sum r5_run.py r5_lib.py g0_wandb.py src/*.py > "$OUT/code.sha256"
/home2/karma/duan-verify/env/bin/python r5_run.py --cells "$CELLS" --bits "$BITS" --out "$OUT" --wandb-name "$NAME-$SLURM_JOB_ID" "$@"
date

#!/bin/bash
# Launch one arm of the G0 run on Ada (u22, 2 x RTX 2080 Ti; model_ref+SAE on cuda:0, model_work on cuda:1).
# Usage: sbatch scripts/g0/run_g0.sbatch <arm-name> <G0_BOS: none|prepend_keep|prepend_mask> [G0_BITS=0,16,12,8,7,6] [G0_TOKENS=500000] [G0_RUN_MODE=g0_core]
# Smoke test:  sbatch scripts/g0/run_g0.sbatch smoke none 8,6 20000 3a_test   (uses the shipped 3a_test mode: 20k tokens, INT8+INT6)
# Prereqs (see PLAN.md): env at /home2/karma/duan-verify/env; `hf auth login` done (token at /home2/karma/duan-verify/hf/token); runners + g0_wandb.py generated/copied next to this script
# NOTE: /share1 is NOT mounted on compute nodes. Everything here lives on /home2 (NFS, visible everywhere) or node-local /scratch.
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J g0
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
ARM=${1:?arm}; BOS=${2:?bos}; BITS=${3:-0,16,12,8,7,6}; TOK=${4:-500000}; MODE=${5:-g0_core}
# --- Hugging Face: the gated model (10.5 GB fp32) is fetched onto NODE-LOCAL /scratch and deleted at the end of the job.
# (/home2 has only ~19 GB free and is shared with other jobs: an earlier attempt nearly filled it. The token stays in /home2.)
export HF_TOKEN_PATH=/home2/karma/duan-verify/hf/token
export HF_HOME=/scratch/karma-g0-hf-${SLURM_JOB_ID:-manual}
export HF_XET_CACHE=$HF_HOME/xet HF_XET_HIGH_PERFORMANCE=1
# NOTE: do NOT set HF_HUB_DISABLE_XET=1: the plain-HTTP path was ~15x slower and kept dropping connections.
mkdir -p "$HF_HOME" /home2/karma/duan-verify/logs
trap 'kill ${MONPID:-0} 2>/dev/null || true; rm -rf "$HF_HOME"' EXIT
for attempt in 1 2 3 4 5 6; do
  /home2/karma/duan-verify/env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json && break
  echo "model download attempt $attempt failed; retrying"; sleep 10
done
du -sh "$HF_HOME"
export G0_RUN_MODE=$MODE G0_BOS=$BOS G0_BITS=$BITS G0_TOKENS=$TOK G0_WORK_DEVICE=cuda:1
export G0_OUT=/home2/karma/duan-verify/out/$ARM
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export WANDB_PROJECT=${WANDB_PROJECT:-anlp-project} G0_WANDB_NAME=${G0_WANDB_NAME:-g0-$ARM-$SLURM_JOB_ID}   # wandb: key comes from ~/.netrc (wandb login); G0_WANDB=0 disables; WANDB_MODE=offline + `wandb sync` as fallback
mkdir -p "$G0_OUT" /home2/karma/duan-verify/logs
nvidia-smi -L; date
# measured GPU memory/utilization every 5 s (answers "does it really fit?")
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 5 > "$G0_OUT/gpu_mem.log" &
MONPID=$!
cd /home2/karma/duan-verify
/home2/karma/duan-verify/env/bin/python duan03_runner_v3.py
date

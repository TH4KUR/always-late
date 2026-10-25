#!/bin/bash
# Llama-3.2-1B consistency check (D-087): CPU-only job on the u22 partition of Ada (no --gres, as for run_gain_statedef.sbatch). Builds the groups manifest (K16 and K64 families) of a finished tagging job.
# Usage: sbatch -J llama-manifest --export=ALL,REPO=/home2/karma/duan-verify/phase1/repo_<sha> scripts/g0/run_llama_manifest.sbatch <tag job name>
# Output: /home2/karma/duan-verify/out/phase1/<tag job name>/groups_manifest.json (read by the lm jobs of run_llama.sbatch).
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -J llama-manifest
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
TAG=${1:?tag job name}
BASE=/home2/karma/duan-verify
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)
OUT=$BASE/out/phase1/$TAG
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs"; date; echo "commit $COMMIT"
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_llama_run.py manifest --selectivity "$OUT/eq/selectivity.pt" --manifest "$OUT/groups_manifest.json"
date; sha256sum "$OUT/eq/selectivity.pt" "$OUT/groups_manifest.json"

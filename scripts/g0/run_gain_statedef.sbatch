#!/bin/bash
# Study G extension B (D-073, spec section 10): CPU-only job on the u22 partition of Ada (no --gres: the CPU partition u22-cpu is not available to this user, as for run_association_analysis.sbatch).
# Builds the state-defined groups manifests (FP as reference, INT6, INT4) from the D-035 per-sentence sidecar, with the 300 evaluation sentences of the main study excluded from every selection.
# Usage: sbatch -J gain-sd-manifests --export=ALL,REPO=/home2/karma/duan-verify/phase1/repo_<sha> scripts/g0/run_gain_statedef.sbatch [extra args for phase1_gain_statedef.py]
# Output: /home2/karma/duan-verify/out/phase1/gain-sd-manifests/{groups_manifest_<state>.json, overlap.csv, diagnostics.csv, meta.json}; the GPU jobs `gain-sd-<state>` read the manifests.
#SBATCH -p u22
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J gain-sd-manifests
#SBATCH -o /home2/karma/duan-verify/logs/%x-%j.out
set -euo pipefail
BASE=/home2/karma/duan-verify
REPO=${REPO:-$BASE/phase1/repo}; COMMIT=$(cat $REPO/COMMIT)
OUT=$BASE/out/phase1/gain-sd-manifests
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs"; date; echo "commit $COMMIT"
cd $REPO/scripts/g0
$BASE/env/bin/python phase1_gain_statedef.py --sidecar $BASE/out/phase1/association-full/association --selectivity $BASE/out/phase1/p1-mask/eq/selectivity.pt \
    --reference-manifest $REPO/reports/functional_validation/groups_manifest.json --out "$OUT" --states FP INT6 INT4 "$@"
date; ls -la "$OUT"; sha256sum "$OUT"/*

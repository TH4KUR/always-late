#!/bin/bash
# Deploy the COMMITTED Phase 1 code to Ada (git archive of HEAD, so what runs is exactly what the tests ran on) and record the commit.
# Usage: bash scripts/g0/deploy_phase1.sh [DEST]      (refuses to deploy with uncommitted changes in the deployed paths)
#   DEST (default `repo`) is a directory name under /home2/karma/duan-verify/phase1/. A job reads the repository it was started from, so deploy a NEW name (for example repo_<short commit>)
#   while jobs run and start the next job with `sbatch --export=ALL,REPO=/home2/karma/duan-verify/phase1/<DEST> ...` (run_functional.sbatch honours REPO); the default `repo` must only be
#   replaced when no job runs.
set -euo pipefail
DEST=${1:-repo}
ADA_SSH=${ADA_SSH:-ssh karma@ada-gw1}                  # override when the alias does not resolve, e.g. ADA_SSH='ssh -o HostKeyAlias=ada-gw1 karma@10.4.24.30' (keeps the host-key check; never accept a changed key)
case "$DEST" in ""|.*|*[!A-Za-z0-9_.-]*) echo "bad DEST name: $DEST (letters, digits, _ . - only, not starting with a dot)"; exit 1;; esac
cd "$(dirname "$0")/../.."
PATHS="scripts/g0 third_party/duan2026/pristine/scripts/03_gemma_phase3_streaming_sweep.py reports/g0/their_functions.json reports/g0/phase1_languages.json reports/g0/baseline_smoke_vendored/phase3_summary_final.csv reports/functional_validation/groups_manifest.json reports/functional_validation/f4 reports/functional_validation/ksens reports/functional_validation/f5 reports/functional_validation/evaluation_freeze_lm.json reports/intervention_transfer/freeze.json reports/intervention_transfer/freeze_external.json reports/intervention_transfer/core_external.json reports/intervention_transfer/core_external_record.json reports/intervention_transfer/core_sample.json reports/intervention_transfer/precision_audit_freeze.json reports/intervention_transfer/precision_audit_aux.json reports/gptq_extension/wikipedia_first_shards.json reports/gptq_extension/calibration_freeze.json reports/gptq_extension/freeze.json"
if [ -n "$(git status --porcelain -- $PATHS)" ]; then echo "uncommitted changes in deployed paths: commit first"; git status --short -- $PATHS; exit 1; fi
if [ "$DEST" = "repo" ] && [ -n "$($ADA_SSH 'squeue -u karma -h')" ]; then echo "jobs are queued or running: deploy to a named snapshot (bash scripts/g0/deploy_phase1.sh repo_<short commit>) instead of replacing 'repo'"; exit 1; fi
COMMIT=$(git rev-parse HEAD)
git archive HEAD $PATHS | $ADA_SSH "set -e; DEST=$DEST; R=/home2/karma/duan-verify/phase1/$DEST; rm -rf \$R; mkdir -p \$R; tar x -C \$R; echo $COMMIT > \$R/COMMIT; mkdir -p /home2/karma/duan-verify/out/phase1; cp \$R/scripts/g0/run_phase1.sbatch \$R/scripts/g0/run_diag.sbatch /home2/karma/duan-verify/; (cd \$R && sha256sum scripts/g0/phase1_run.py scripts/g0/phase1_lib.py scripts/g0/phase1_data.py scripts/g0/their_code.py)"
echo "deployed $COMMIT"

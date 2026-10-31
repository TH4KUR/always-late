#!/bin/bash
# Wrapper for jarvis_run_all.sh: pre-seeds legacy tier-1 from HF,
# runs setup+gate (Phase A), then full three-arm two-tier run (Phase B).
#
# Usage:
#   bash scripts/g0/jarvis_budget_run.sh          # real run
#   DRY_RUN=1 bash scripts/g0/jarvis_budget_run.sh  # prints plan only
#
# Requires: scripts/g0/.env with HF_TOKEN (and optionally WANDB_API_KEY).
# Amendment A2 is written before any driver call.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BASE=${BASE:-$(dirname "$REPO")}
DRY_RUN=${DRY_RUN:-0}
START_EPOCH=$(date +%s)

STATE_DIR=$BASE/state
OUTROOT=$BASE/out/phase1
LOGS=$BASE/logs
HASHES=$REPO/reports/functional_validation/gemma9b/artifact_hashes.json
AMEND_FILE=$REPO/reports/functional_validation/gemma9b/amendments.md

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
elapsed() { echo $(( $(date +%s) - START_EPOCH )); }

# ── 0. Source env file ─────────────────────────────────────────────────────────
ENV_FILE=${ENV_FILE:-$HERE/.env}
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
    say "sourced $ENV_FILE"
else
    say "no env file at $ENV_FILE (set ENV_FILE or export HF_TOKEN directly)"
fi

[ "$DRY_RUN" = 1 ] || [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set: create scripts/g0/.env with HF_TOKEN=hf_... or export it"
export HF_HOME=${HF_HOME:-$BASE/hf_cache}

# ── 1. Pre-flight checks ──────────────────────────────────────────────────────
say "=== Pre-flight ==="

if [ "$DRY_RUN" != 1 ]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    say "NVIDIA driver: $DRIVER"

    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    say "GPU VRAM: ${VRAM_MIB} MiB"
    [ "${VRAM_MIB:-0}" -ge 38000 ] || die "GPU VRAM ${VRAM_MIB} MiB < 38000 MiB minimum"

    RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
    say "Host RAM: ${RAM_GB} GB"

    DISK_FREE_GB=$(df --output=avail "$BASE" 2>/dev/null | tail -1 | awk '{printf "%d", $1/1048576}')
    say "Free disk: ${DISK_FREE_GB} GB"
    [ "${DISK_FREE_GB:-0}" -ge 60 ] || say "WARNING: only ${DISK_FREE_GB} GB free disk (need ~80 GB)"

    BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
    say "Branch: $BRANCH"
fi

# ── 2. Pre-seed legacy tier-1 from HF ─────────────────────────────────────────
preseed_legacy() {
    if [ -f "$STATE_DIR/legacy_preseeded.done" ]; then
        say "[skip] legacy pre-seed (already done)"
        return 0
    fi
    say "=== Pre-seeding legacy tier-1 outputs from HF ==="

    local HF_DATASET="karma-skz/always-late-artifacts"

    mkdir -p "$STATE_DIR" "$OUTROOT"

    local dl_script
    dl_script=$(cat <<'PYEOF'
import os, sys, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

dataset = sys.argv[1]
outroot = Path(sys.argv[2])
state_dir = Path(sys.argv[3])

cache = snapshot_download(
    dataset, repo_type="dataset",
    allow_patterns=["gemma9b/runs/full/*"],
    token=os.environ.get("HF_TOKEN")
)
src = Path(cache) / "gemma9b" / "runs" / "full"
if not src.is_dir():
    print(f"ERROR: {src} not found in downloaded dataset", file=sys.stderr)
    sys.exit(1)

dirs_to_copy = ["g9-tag", "g9-lm-FP", "g9-lm-4", "g9-lm-6", "g9-lm-8", "g9-lm-7", "g9-lm-5"]
for d in dirs_to_copy:
    s = src / d
    t = outroot / d
    if not s.is_dir():
        print(f"WARNING: {s} not in HF dataset, skipping", file=sys.stderr)
        continue
    if t.exists():
        shutil.rmtree(t)
    shutil.copytree(str(s), str(t))
    print(f"  copied {d} ({sum(1 for _ in t.rglob('*') if _.is_file())} files)")

for d in dirs_to_copy:
    t = outroot / d
    if not t.is_dir():
        print(f"ERROR: {d} missing after copy", file=sys.stderr)
        sys.exit(1)

tag_sel = outroot / "g9-tag" / "eq" / "selectivity.pt"
if not tag_sel.exists():
    print(f"ERROR: {tag_sel} missing", file=sys.stderr)
    sys.exit(1)

for s in ["FP", "4", "6", "8", "7", "5"]:
    lm_dir = outroot / f"g9-lm-{s}"
    pts = list(lm_dir.glob("func_lm_*_*.pt"))
    if not pts:
        print(f"ERROR: no func_lm_*.pt in {lm_dir}", file=sys.stderr)
        sys.exit(1)

for s in ["FP", "4", "6", "8", "7", "5"]:
    (state_dir / f"state_{s}.done").touch()
    print(f"  marker: state_{s}.done")

(state_dir / "legacy_preseeded.done").touch()
print("Legacy pre-seed complete")
PYEOF
    )

    if [ "$DRY_RUN" = 1 ]; then
        echo "+ pre-seed: download $HF_DATASET/gemma9b/runs/full -> $OUTROOT/g9-{tag,lm-FP,lm-4,...}"
        echo "+ pre-seed: create markers state_{FP,4,6,8,7,5}.done"
        return 0
    fi

    local py=python3
    [ -x "$BASE/venv/bin/python" ] && py=$BASE/venv/bin/python

    if ! "$py" -c "import huggingface_hub" 2>/dev/null; then
        say "installing huggingface_hub for pre-seed download..."
        "$py" -m pip install --quiet huggingface_hub 2>/dev/null || {
            python3 -m pip install --quiet --user huggingface_hub
            py=python3
        }
    fi

    "$py" -I -c "$dl_script" "$HF_DATASET" "$OUTROOT" "$STATE_DIR" || die "legacy pre-seed from HF failed"
    say "legacy pre-seed done ($(elapsed)s elapsed)"
}

# ── 3. Write amendment A2 before any driver call ───────────────────────────────
write_amendment() {
    if [ -f "$AMEND_FILE" ]; then
        say "[skip] amendments.md already exists"
        return 0
    fi
    [ "$DRY_RUN" = 1 ] && { echo "+ write amendments.md"; return 0; }
    cat > "$AMEND_FILE" <<'AEOF'
# Amendments to the Gemma-2-9B confirmation run

## Amendment A2: Legacy reuse from HF (2026-10-30)

**What:**
1. Legacy tier-1 outputs reused from HF dataset `karma-skz/always-late-artifacts`
   (folder `gemma9b/runs/full`, produced with adapter hash 8cc54217).
   Manifest rebuilt with current adapter (0a69b263). Only `all` family Rs differs
   ((5,5,5) to (5,5,3)); K64 groups identical. Predictions use K64.
2. PROCS_PER_GPU=2 for parallel new-arm tier-1 units.
3. INT7/INT5/INT8 included via HF download (not re-computed).

**Why:** Legacy tier-1 already completed and verified (verdict: Consistent).
Reuse saves ~2.5 hours of GPU time (tagging + 6 legacy states).

**What is NOT changed:**
- freeze.json, amendment_A1_three_arm.json: unchanged
- All frozen code files: unchanged
- Calibration pool, seeds, languages, sentence counts, ratios, thresholds: unchanged
- jarvis_run_all.sh: called as-is via wrapper, never edited

**Impact on predictions:**
- P1-P3b (legacy arm): met per verdict.md (pre-seeded outputs)
- Q0-Q3 (GPTQ/RTN-matched comparison): evaluable when tier-1 completes
- S1-S4 (tier-2): evaluable when tier-2 completes

**Outcome:** (filled in after completion)
AEOF
    say "wrote $AMEND_FILE"
}

# ── 4. Phase A: setup + validation + gate ──────────────────────────────────────
phase_a() {
    say "=== Phase A: setup + smokes + gate (STOP_AFTER=smoke_gptq) ==="
    say "elapsed: $(elapsed)s"

    if [ "$DRY_RUN" = 1 ]; then
        echo "+ FOREGROUND=1 STOP_AFTER=smoke_gptq STAGES=\"g p1\" PROCS_PER_GPU=2 bash scripts/g0/jarvis_run_all.sh"
        return 0
    fi

    ( cd "$REPO" && \
        FOREGROUND=1 \
        STOP_AFTER=smoke_gptq \
        STAGES="g p1" \
        PROCS_PER_GPU=2 \
        bash scripts/g0/jarvis_run_all.sh
    ) || {
        local rc=$?
        say "Phase A failed (exit $rc)"
        tail -40 "$BASE/run_all.log" 2>/dev/null || true
        die "Phase A did not complete: check $BASE/run_all.log and per-step logs in $LOGS/"
    }

    say "Phase A complete ($(elapsed)s elapsed)"
}

# ── 5. Phase B: full three-arm two-tier run ───────────────────────────────────
phase_b() {
    say "=== Phase B: full run (STAGES=g p1, all arms) ==="
    say "elapsed: $(elapsed)s"

    # Neutralize full.done so driver enters do_full
    if [ "$DRY_RUN" != 1 ]; then
        rm -f "$STATE_DIR/full.done"
    fi

    if [ "$DRY_RUN" = 1 ]; then
        echo "+ FOREGROUND=1 STAGES=\"g p1\" ARMS=\"legacy rtn-matched gptq\" PROCS_PER_GPU=2 bash scripts/g0/jarvis_run_all.sh"
        return 0
    fi

    ( cd "$REPO" && \
        FOREGROUND=1 \
        STAGES="g p1" \
        ARMS="legacy rtn-matched gptq" \
        PROCS_PER_GPU=2 \
        bash scripts/g0/jarvis_run_all.sh
    ) || {
        local rc=$?
        say "Phase B finished with exit $rc (some units may have failed)"
        say "Check $BASE/run_all.log and unit logs in $LOGS/"
    }

    say "Phase B complete ($(elapsed)s elapsed)"
}

# ── 6. Report ─────────────────────────────────────────────────────────────────
report() {
    say "=== Final Report ==="
    local total_sec=$(elapsed)
    local total_min=$(( total_sec / 60 ))

    say "Total wall time: ${total_min} min (${total_sec}s)"

    local done=0 failed=0
    for f in "$STATE_DIR"/state_*.done "$STATE_DIR"/unit_*.done; do
        [ -f "$f" ] && done=$((done + 1))
    done
    for f in "$STATE_DIR"/unit_*.failed "$STATE_DIR"/state_*.failed; do
        [ -f "$f" ] && failed=$((failed + 1))
    done

    say "Units: $done done, $failed failed"

    say "Completed:"
    for f in "$STATE_DIR"/state_*.done "$STATE_DIR"/unit_*.done; do
        [ -f "$f" ] && say "  $(basename "$f" .done)"
    done

    if [ "$failed" -gt 0 ]; then
        say "Failed:"
        for f in "$STATE_DIR"/unit_*.failed "$STATE_DIR"/state_*.failed; do
            [ -f "$f" ] && say "  $(basename "$f" .failed)"
        done
    fi

    local has_gptq_4=0 has_gptq_6=0 has_rtnm_4=0 has_rtnm_6=0
    [ -f "$STATE_DIR/unit_g_gptq_4.done" ] && has_gptq_4=1
    [ -f "$STATE_DIR/unit_g_gptq_6.done" ] && has_gptq_6=1
    [ -f "$STATE_DIR/unit_g_rtn-matched_4.done" ] && has_rtnm_4=1
    [ -f "$STATE_DIR/unit_g_rtn-matched_6.done" ] && has_rtnm_6=1

    say ""
    say "Evaluable predictions:"
    say "  P1-P3b (legacy arm): YES (pre-seeded from HF)"
    if [ "$has_gptq_4" = 1 ] && [ "$has_gptq_6" = 1 ] && [ "$has_rtnm_4" = 1 ] && [ "$has_rtnm_6" = 1 ]; then
        say "  Q0-Q3 (tier-1 three-arm): YES"
    else
        say "  Q0-Q3 (tier-1 three-arm): PARTIAL (gptq: 4=$has_gptq_4 6=$has_gptq_6, rtnm: 4=$has_rtnm_4 6=$has_rtnm_6)"
    fi

    local has_p1_gptq=0 has_p1_legacy=0 has_p1_rtnm=0
    [ -f "$STATE_DIR/unit_p1_gptq.done" ] && has_p1_gptq=1
    [ -f "$STATE_DIR/unit_p1_legacy.done" ] && has_p1_legacy=1
    [ -f "$STATE_DIR/unit_p1_rtn-matched.done" ] && has_p1_rtnm=1

    if [ "$has_p1_gptq" = 1 ] && [ "$has_p1_legacy" = 1 ] && [ "$has_p1_rtnm" = 1 ]; then
        say "  S1-S4 (tier-2): YES"
    elif [ "$has_p1_gptq" = 1 ] || [ "$has_p1_legacy" = 1 ]; then
        say "  S1-S4 (tier-2): PARTIAL (gptq=$has_p1_gptq legacy=$has_p1_legacy rtnm=$has_p1_rtnm)"
    else
        say "  S1-S4 (tier-2): NO (tier-2 not run or not complete)"
    fi

    say ""
    if [ -f "$BASE/gemma9b_results.tar.gz" ]; then
        say "Results tarball: $BASE/gemma9b_results.tar.gz ($(du -h "$BASE/gemma9b_results.tar.gz" | cut -f1))"
        say "SCP command:  scp $(whoami)@\$(hostname):$BASE/gemma9b_results.tar.gz ."
    fi
    [ -f "$HASHES" ] && say "Artifact hashes: $HASHES"
    say ""
    say ">>> PAUSE THE VM NOW to stop billing <<<"
}

# ── Main ───────────────────────────────────────────────────────────────────────
say "========================================================"
say "  Gemma-9B full three-arm two-tier run"
say "  $(date)"
say "========================================================"

[ "$DRY_RUN" = 1 ] && say "DRY RUN — printing plan only"

preseed_legacy
write_amendment
phase_a
phase_b
report

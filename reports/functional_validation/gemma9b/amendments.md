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

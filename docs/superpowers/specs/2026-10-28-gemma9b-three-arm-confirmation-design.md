# Gemma-2-9B as a second-model confirmation of the project as it now stands (three quantizer arms): design

Date: 2026-10-28 IST. Author: Claude, at the user's request of the same day ("keep th4kur's scripts and changes, make updates to that so that the 9b runs are confirmatory for our updates as well and not just the ones we planned at that time"; "i at least want to do the runs using GPTQ now, and not RTN"). The user approved the design in chat on 2026-10-28. Builds on D-089 (Gemma-2-9B replaces Llama), `docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md` and the work of Shardul0007 (adapter, freeze, driver) and th4kur (VM setup script), all kept. Branch `gemma9b-confirmation-v2` (main + both 9B branches merged; their branches are untouched).

## 1. What is wanted

The 9B run on Jarvis Labs must confirm (or fail to confirm) what the project claims NOW, not only the legacy-RTN Study G of 2026-10-26. Since then the project added two quantizers (matched RTN and GPTQ, D-101 to D-110) and found that **the quantizer changes the size of the effects, mostly not their direction** (`reports/gptq_extension/RESULTS.md` sections 1 and 5): GPTQ shrinks the 4-bit amplification of the loss caused by removing a language's features (Study G, deepest down ratio: 1.9 times the full-precision harm with legacy RTN, 1.7 matched RTN, 1.1 GPTQ), leaves the 6-bit picture unchanged, removes the "boosting repairs" hint, and makes the Phase 1 null fragile. These are the claims the second model should test.

Constraints: no frozen file is edited (the 9B freeze pins `phase1_gemma9b_run.py`, `run_gemma9b.sbatch`, its test and the roster; the old entry points are pinned by the 2B freezes); keep th4kur's and Shardul0007's scripts (extended, not replaced); everything frozen BEFORE any 9B number exists (a dated amendment); a result that disagrees is a valid result; nothing is run on Jarvis until the user says so; main branch and the cron of the 2B work are not disturbed.

## 2. Machine (given by the user, 2026-10-28)

1x RTX PRO 6000 (Blackwell, compute capability 12.0), 96 GB VRAM, 28 vCPU, 160 GB RAM, 250 GB SSD, India region, about 210 INR per hour. One GPU: states and arms run one after the other (or two processes side by side if the smoke job shows the GPU is not saturated and RAM allows: `PROCS_PER_GPU`). Facts that follow: (a) torch 2.14.1 (the pinned version of the 2B study) has wheels only for CUDA 12.6 and 13.0, and sm_120 needs the 13.0 build (driver 580 or newer), so the driver chooses the index from the compute capability and refuses with a clear message when the driver is too old; (b) the 9B model in bf16 is 18.5 GB, so it sits on the single GPU (`GEMMA9B_N_DEVICES=1`, the frozen loader unchanged); (c) a GPTQ or matched-RTN weight set of the target matrices is about 17.6 GB in bf16 on the CPU, two of them plus the full-precision copy and the calibration transients are about 75 GB of the 160 GB RAM, so one process holds at most the two artifacts (INT6, INT4) of one arm.

## 3. Scope

- **Tier 1, Study G (the frozen core).** The 9B Study G MAIN preset (27 languages, K64 groups, 3 matched-control draws, 300 FLORES devtest sentences, the 2B ratios, tagging on dev) in three arms: **legacy RTN** exactly as frozen (states FP, INT8, INT7, INT6, INT5, INT4; predictions P1 to P3b unchanged), **matched RTN** and **GPTQ** (states INT6 and INT4, the states of the 2B extension; the FP file is the legacy arm's). Tagging and the manifest are done once, in full precision, and shared by the arms.
- **Tier 2, Phase 1 survival and loss (the original pre-set tests T1 to T3 and the loss ladder).** `phase1_run:main` (mask regime, FLORES) in the same three arms through the same wrapper: legacy whole ladder (8 7 6 5 4 16 12 0), matched RTN and GPTQ at bits 7 6 4 and the floors 16 12 0 (the bits T1 to T3 need, plus the 4-bit level). Both tiers are in scope (the user, 2026-10-28: "both tier 1 and 2 are important. prepare for both"); tier 2 runs after tier 1 in the same session. The analysis is the unchanged `phase1_analysis.py` per arm, `gptq_ext/compare_stage1.py` across arms, and the frozen predictions S1 to S3 of section 5.
- **Not in this round:** the transfer study and the external panel (they need a 9B freeze and a new feature core), E4/E5 (rescue), Belebele, F5, the association study, the precision audit.

## 4. The GPTQ arms on 9B (design)

The adapter is not edited. `gptq_ext/run_with_weights.py` wraps any entry point; for 9B a new thin module `phase1_gemma9b_ext.py` (new file, imports the adapter's `gemma9b_setting`) provides the entry point `phase1_gemma9b_ext:main` with the tasks `lm` (delegates to the adapter's frozen `main`) and `p1` (runs `phase1_run.main` inside `gemma9b_setting`, tier 2). Both are called through `run_with_weights.py --entry phase1_gemma9b_ext:main --weights gptq|rtn-matched`, which prebuilds the artifacts of the bit widths in `--precisions`/`--bits` right after the model load, refuses an artifact whose sha256 differs from the frozen one, and writes `weights_manifest.json`.

Needed for 9B, each with a test first:
1. **Calibrator on 9B shapes.** The canonical adapter (`canonical.py`) and the calibrator read shapes from the model; a CPU toy model with the 9B layout (grouped-query attention with fewer key/value heads, d_head different from d_model / heads, post-norms, tied embeddings) must give: identity method = original activations at every stage, matched RTN equal to a direct computation, GPTQ error not worse than matched RTN on the stage loss.
2. **Same tokenizer, same pool.** The frozen calibration pool is tokens of the Gemma-2-2B tokenizer; Gemma-2-9B shares the tokenizer. A check compares the `tokenizer.json` hashes of the two models (and the token ids of a probe string) before the pool is used; a mismatch stops the run. The pool token files (221,184 tokens, about 1 MB) are shipped with the repository bundle for Jarvis (not in git) and verified against `reports/gptq_extension/calibration_freeze.json` (pool sha256 `5432345939...`).
3. **9B artifact hashes.** The 2B hashes in `reports/gptq_extension/artifact_hashes.json` do not apply. A build-and-gate job on Jarvis (`gptq_ext.gate_run` with the 9B adapter, `--record-weights-hashes`) builds the INT6 and INT4 artifacts of both methods, runs the gates (targets only changed, restore exact, FP logits bit-identical after all switches, a second build of the first artifact equal) and writes `reports/functional_validation/gemma9b/artifact_hashes.json`; every later job runs with `FROZEN=1` and refuses a different hash. The hashes are recorded on this GPU type; equality on other hardware is not claimed (stated in the report).
4. **Memory guard.** A pre-job check refuses to start a process whose estimate (FP copy + artifacts + transients, 18 GB per artifact) exceeds the RAM left, naming the numbers.
5. **Driver.** `jarvis_run_all.sh` (Shardul0007's, kept) is extended in place: arms (`ARMS="legacy rtn-matched gptq"`), stages (`STAGES="g"`, `g p1`), the Blackwell-aware torch index, the pool and tokenizer check, a gate step before the GPTQ arms, `PROCS_PER_GPU`, the smoke test extended to a GPTQ INT4 build on 3 languages, resumable per (arm, stage, state) as before, packaging of the new outputs. Its dry-run test is extended; the old tests keep passing.

## 5. Predictions (frozen before any 9B number; a dated amendment `amendment_A1_three_arm.json` with the sha256 of this section's text, committed before the run)

P1 to P3b (legacy arm) are the frozen ones, unchanged (`freeze.json`, predictions text sha256 `6da85e5b...`), with the 2026-10-26 verdict function. New, for the quantizer comparison, using the frozen analysis code of Study G on each arm (`phase1_gain_analysis`, `phase1_llama_compare.evaluate`), the tested languages (group of at least 16 members), `req(x) = ceil(x*n/27)`; thresholds from the 2B arms (D-107, D-109):

- **Q0 (existence and grading hold under the better quantizers):** in the matched-RTN arm and in the GPTQ arm, P1 holds at FP (the FP file is shared, so once) and P3a (H-G1) holds at INT4. (2B: met in both arms.)
- **Q1 (the quantizer ordering at INT4):** let r(arm) be the median over the tested languages with a positive FP value of net harm at the deepest down ratio (0.0884), INT4 over FP. Q1 holds if r(GPTQ) <= r(matched RTN) and r(GPTQ) <= r(legacy) - 0.3, and the per-language ratio is lower under GPTQ than under legacy in at least req(20) languages. Evaluated only if the legacy arm itself shows an amplification (P3b: median ratio above 1.3); otherwise it is reported as not evaluable, not as failed. (2B: legacy 1.9, matched 1.7, GPTQ 1.1.)
- **Q2 (6 bits are the same in every arm):** the macro retention at INT6 lies in [0.85, 1.15] in all three arms. (2B: yes.)
- **Q3 (the repair hint, reported, never part of the verdict):** H-G5b (a group-specific repair at ratio 1.1 or 1.2, INT4) per arm, and the number of languages whose best fine ratio is above 1. (2B: H-G5b not met under matched RTN and GPTQ; best factor above 1 in 21, 13 and 7 of 27 languages for legacy, matched, GPTQ.)

**Verdict for the project's updates (the complete function):** *consistent* = Q0 and Q1 and Q2; *partly consistent* = Q0 and Q2 and not Q1 (Q1 failed, or not evaluable because the legacy arm shows no amplification: the report says which); *different* = not Q0 or not Q2. P2a and Q3 are reported with their outcomes and never change the verdict. Wording kept to what is tested: a scale replication within one architecture and one SAE family; no claim about other architectures, layers or SAEs; all contrasts are per-arm (no paired contrast across arms is a prediction, except the per-language count inside Q1); the 2B numbers are the committed ones.

**Tier 2 (Phase 1 survival and loss), frozen the same way.** For each arm the unchanged `phase1_analysis.py` gives T1 to T3 and the per-language survival and loss increase (`compare_stage1.load_arm`: survival = share of active features with r > 0.9, loss increase = nats per token over FP, equal-budget stream `eq/`); tested languages = the 27 of the roster. Thresholds from the 2B arms (D-104, D-106):

- **S1 (survival ordering):** the mean survival over languages at INT6 and at INT4 is ordered GPTQ > matched RTN > legacy RTN at both bit widths, and GPTQ is above legacy RTN in at least req(24) languages at each. (2B: INT6 97.5 / 89.7 / 85.0 %, INT4 30.5 / 18.7 / 12.7 %; GPTQ above legacy in 27 of 27.)
- **S2 (loss ordering):** the mean loss increase over languages at INT6 and at INT4 is ordered GPTQ < matched RTN < legacy RTN at both bit widths, and GPTQ is below legacy in at least req(24) languages at each. (2B: INT6 0.0058 / 0.0197 / 0.0254, INT4 0.130 / 0.385 / 0.590.)
- **S3 (the tier gap at 6 bits belongs mostly to the legacy quantizer):** high-resource minus low-resource mean survival at INT6 (the roster's tiers) is smaller under GPTQ than under legacy RTN. (2B: the gain of GPTQ over legacy is +25 points for the low tier and +4 for the high tier.)
- **S4 (the project's original tests T1 to T3, reported, no direction predicted):** the verdict word of the three-test family per arm under the original rules (Holm over three), and whether the arms agree. (2B: legacy null, matched RTN crosses 0.05 at 7 and 6 bits, GPTQ null; the project reads this as fragile, not as a result.)

**Tier 2 verdict:** *consistent* = S1 and S2 and S3; *partly consistent* = S1 and S2 and not S3; *different* = not S1 or not S2. S4 is reported with its outcome and never changes the verdict.

**Both tiers together** are reported as two verdicts, never merged into one word.

(The Q0 to Q3 list, the S1 to S4 list and the two verdict paragraphs are the amendment's `text`; its sha256 is stored in the amendment.)

## 6. Analysis tool (tests first)

`phase1_gemma9b_compare.py` (CPU): tier 2 part: loads the three arms' `survival.pt` with `compare_stage1.load_arm`, computes S1 to S3 (S4 comes from `phase1_analysis.py` per arm, summarised) and the tier-2 verdict. Tier 1 part: loads the runs of each arm with `ga.load_runs`, calls `phase1_llama_compare.evaluate` per arm (P1 to P3b and the legacy verdict; for the two new arms with the states FP, INT6, INT4 P2a is restricted to INT6 and reported as such), computes Q0 to Q3 across the arms, reads the 2B reference numbers of the committed outputs (legacy `gain_study/results/curves.csv`, `gptq_extension/results/gain/{rtnm,gptq}/curves.csv`) and writes `analysis.json` and `report.md`. Tested on synthetic run files; the 2B outputs are run through it as a control (the 2B legacy, matched and GPTQ arms must give the known numbers: H-G1 met, ratios 1.9 / 1.7 / 1.1).

## 7. Gates on Jarvis (instrument gates, not looks at results)

(1) GPU line and CUDA check, torch import on sm_120; (2) the token reaches `google/gemma-2-9b`; (3) tokenizer equality with 2B; (4) pool and freeze hashes; (5) smoke: tag + manifest on 3 languages and `lm` FP and INT4 for the legacy arm (the old smoke), then the tier-2 path on the same 3 languages (`p1`, legacy rounding, 4 bits); (6) the build-and-gate job (section 4.3), then a GPTQ INT4 `lm` state on the 3 smoke languages through the wrapper with the recorded hashes (proves the whole GPTQ path before the long units); GPU memory and seconds per step are printed throughout. Any failure stops the run and is reported; amendments are written before a re-run.

## 8. Risks and how they are handled

- The adapter, the calibrator and the new wrapper were never run on the real 9B model: the smoke job and the gate job are the first real execution (stated in the hand-over).
- Blackwell + torch 2.14.1 cu130 + TransformerLens/sae-lens pins: untested; the env step prints the torch/CUDA/device line and runs a bf16 matmul and a SAE-free forward on a tiny tensor before downloads.
- GPTQ build time on 9B is unmeasured (2B: about 20 minutes per artifact on a 2080 Ti); the gate job prints it, the plan's drop order (section 9) applies.
- RAM: section 4.4.
- Budget: about 210 INR per hour; the plan lists the GPU hours per step as measured by the smoke job, tier 2 is optional.

## 9. Order and drop order

smoke -> tag + manifest -> legacy arm FP -> GPTQ gate/hash recording -> tier 1: GPTQ INT4, INT6 -> matched RTN INT4, INT6 -> legacy INT4, INT6, INT8, INT7, INT5 -> tier 2: Phase 1 for GPTQ, legacy, matched RTN. Both tiers are wanted; if time or money runs out anyway, the order of dropping is: legacy INT7/INT5/INT8 first, then the tier-2 matched-RTN arm, never GPTQ, legacy FP/INT6/INT4, the tagging, or the tier-2 legacy and GPTQ arms (S1 to S3 need those two and matched RTN; without matched RTN the ordering claims are reduced to GPTQ against legacy and reported so).

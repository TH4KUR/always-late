# Detailed analysis of all results of the "Always Late" project (independent audit)

Written 2026-10-29 (IST) by Claude (Opus 5.5) as a **read-only audit**. No experiment, job or analysis of the project was run or changed. The only new computations are small CPU re-computations from committed result files, made to check the numbers. The scripts are in `/tmp/opus_analysis/`, outside the repository. Every number below is given with its source file, relative to `reports/` unless the path says otherwise. Where a number was only re-read from a generated report and not recomputed, this is said.

**How to read the evidence labels used throughout**

| label | meaning in this document |
|---|---|
| **ESTABLISHED** | Large, consistent and controlled within the stated setting (Gemma-2-2B, layer 12, Gemma Scope 16k, simulated weight-only quantization, 27 FLORES languages). Robust to the checks the project ran. Still exploratory if the project labelled it so. |
| **SUGGESTIVE** | A real-looking signal that is small, post hoc, under-powered, fragile to analysis choices, or not reproduced. |
| **NULL** | Tested properly (adequate design, positive controls or equivalence logic) and no effect was found. |
| **INCONCLUSIVE** | Under-powered, ambiguous, or the verdict depends on an analysis convention. |
| **NOT SUPPORTED** | The claim was tested, and the data argue against it or fail the stated criterion. |
| **NOT TESTED** | No experiment in the repository addresses it. |
| **UNVERIFIED** | Results imported from another branch or team and not yet independently reproduced (Gemma-2-9B). |

"Confirmatory" and "exploratory" follow the project's own labels. Section 6.1 explains where those labels themselves are shaky.

---

## 0. Executive summary

1. **The project's starting hypothesis is not supported.** The hypothesis was that weight quantization selectively destroys language-specific SAE features, especially for low-resource languages. The pre-specified Phase 1 tests are null under the original quantizer. The selectivity gap is about 1.4 percentage points at INT6 and reverses once features are matched on activation size. The low-resource survival gap (69 / 91 / 95 % at INT6) disappears when activation scale is added as a covariate. The gap also exists in a threshold-free measure at INT16, where the weights barely change. It nearly closes under GPTQ. Status: **NOT SUPPORTED** as a language-specific quantization effect.
2. **Language-feature groups are functionally real.** Zero-ablating a language's 64 most selective features raises that language's next-token loss far beyond matched controls in 27 of 27 languages. The mean excess is 0.704 [0.684, 0.723] nats/token and the median 0.262. This dependence is kept through INT5. Under legacy RTN it is inflated at INT4: the median ratio is 1.54, falling to 1.36 under matched RTN and 1.22 under GPTQ. Status: **ESTABLISHED** in this setting. The INT4 inflation is **ESTABLISHED, but its size depends on the quantizer**.
3. **The only confirmatory downstream test is null.** On Belebele (18 languages, 385 items each), removing the group does not change accuracy beyond controls: 0.426 / 0.426 / 0.425. A positive control shows the instrument can detect edits of 1 to 6 accuracy points. The same null holds for the whole selective group and for the GPTQ INT4 model, which scores above chance. Status: **NULL** for effects larger than roughly half an accuracy point.
4. **Survival and association metrics are poor proxies for function.** Five lines of evidence agree:
   - the association tag predicts functional continuity at AUROC 0.506;
   - the continuous activation correlation improves the prediction of effect similarity by only 3–4 % over baselines, against a 10 % target, on two models;
   - gain and threshold-unit perturbation add nothing beyond that;
   - most functional drift is shared with control directions;
   - the survival correlation scales with feature size (Spearman about −0.5 with the spread, −0.7 with the peak).

   Status: **ESTABLISHED** as a negative or measurement result in this setting. This is the most coherent and most defensible body of evidence in the project.
5. **Where quantization damage "sits" was not identified.** Patching the full-precision activations of the group back in repairs 2.6 % of the INT4 loss gap. Patching the whole layer-12 residual repairs 39 %. These are repair effects of particular patches, not causal shares. Status: the measurement is **ESTABLISHED**. Any localisation claim is **NOT TESTED / not identifiable** with this design.
6. **Generalisation is weak.** The Llama-3.2-1B check gave the pre-set verdict "not consistent". It is confounded by model, SAE architecture, recipe and depth. The Gemma-2-9B results (original RTN only) report "consistent". They are imported from another branch and still **UNVERIFIED**. Their raw data, per-state metadata and two prediction definitions (P3a, P3b) are unresolved (section 4.14).
7. **What is publishable now** is a measurement-validity and negative-result paper: survival metrics versus intervention effects under three quantizers, with honest nulls. It would still need a document-held-out or second-model validation and a real-kernel check to be robust (section 9).

---

## 1. The setting

| element | value | source |
|---|---|---|
| model | Gemma-2-2B (bf16 body); second models Llama-3.2-1B (exploratory) and Gemma-2-9B (imported, unverified) | `PROJECT_RESULTS.md` §1; `functional_validation/llama/`; `functional_validation/gemma9b/` |
| SAE | Gemma Scope 16,384 features, `layer_12/width_16k/canonical`, `blocks.12.hook_resid_post` (JumpReLU) | `functional_validation/RESULTS_SUMMARY.md` §1 |
| quantizers | legacy RTN (the project's original, with a head-shared scale and its own grid), matched RTN (per output channel, GPTQ grid), GPTQ (Wikipedia calibration, 8,192 tokens × 27 languages) | `gptq_extension/RESULTS.md` §2 |
| bit widths | INT8, 7, 6, 5, 4; INT16/INT12 as near-no-change references; INT0 as restore-only identity | `phase1/mask/eq/summary.csv` |
| data | FLORES-200: dev (997 sentences) for tagging, devtest (1,012) for evaluation; 27 languages, 9 per tier | `g0/phase1_languages.json`, D-010 |
| regime | `mask`: one BOS prepended and excluded from every statistic (D-001, D-027, D-028) | `audit_our_vs_duan.md` |
| quantization type | simulated (fake-quant) weight-only rounding, executed in bf16/float; no integer kernels, no activation quantization | `CURRENT_STATUS.md`, `PROJECT_RESULTS.md` §1 |

---

## 2. The research questions and how they changed

### 2.1 Original questions (proposal `anlp_revised.txt`, lines 28–35 and 249–298)

| id | question | final status |
|---|---|---|
| RQ1 | Do language-specific SAE features survive quantization worse than other features? | **NULL** under the pre-specified tests (legacy RTN); the significance verdict is fragile across quantizers (section 4.2, 4.11) |
| RQ2 | Does survival of language-specific features depend on the language's resource level? | The selective-minus-control gap **NULL with equivalence** (TOST within ±0.02). The overall tier gap in survival is **ESTABLISHED as a descriptive association**, but it is **NOT SUPPORTED as language-specific quantization damage**: it is explained by activation scale, mostly specific to legacy RTN at INT6, and present at INT16 in λ |
| RQ3 | Does geometric feature survival predict function/behaviour? (Phase 2 "causal check") | **NOT SUPPORTED** as a useful predictor: tag AUROC about 0.5; R1 adds 3–4 %; drift is mostly generic |

### 2.2 How the questions evolved (with the recorded reasons)

| date | change | reason recorded | D-entries |
|---|---|---|---|
| 15–17 Oct | Replicate Duan (2026) with their own vendored code; our first pipeline retired | Our pipeline disagreed; their code reproduced their table within 0.15 pp | D-002, D-009 |
| 17–21 Oct | No-BOS regime abandoned for `mask` | Without BOS, FLORES perplexity is 48,864 and the noise-floor gate fails (A-029). With BOS masked, WikiText INT6 survival is 97.5 % instead of 51 % | D-001, D-027, D-028, A-029, A-030 |
| 21 Oct | Publishability made the main objective; questions may change with documented reasons | User instruction | D-029 |
| 21 Oct | Survival detection to be reassessed; INT5/INT4 added | Phase 1 null, magnitude confound | D-030, D-031, D-032–D-034 |
| 21 Oct | Association retention across 27 languages | User selection | D-035–D-038 |
| 21–23 Oct | Functional study F1–F5 + Belebele as the only confirmatory family | Association ≠ function; need a causal check | D-039–D-064 |
| 24–26 Oct | Study G (graded gain), made "the major study" by the user | User instruction | D-070–D-085 |
| 26 Oct | Second model: Llama-3.2-1B, then Gemma-2-9B chosen as the proper confirmation | Llama confounds model and SAE type | D-087, D-089, D-090, D-092 |
| 26–27 Oct | Intervention transfer (context vs weights, Codex's design) + recovery analyses R1–R4 | Novelty search: broad claims already covered | D-088, D-094–D-099 |
| 27 Oct | HOLD; then lifted for one purpose: GPTQ / matched-RTN extension | User request for a fair quantizer comparison | D-101–D-111 |
| 28 Oct | Survival versus effect (post hoc) | Codex's "publication rescue" plan, validity-checked | D-110–D-114 |
| 29 Oct | Paper direction: measurement validity of survival (λ decomposition, ν in threshold units) | Survival rates track size and quantizer, not function | D-117–D-124 |
| 29 Oct | Gemma-2-9B legacy-RTN results imported; CPU review pending | User request | D-125 |

**Assessment of the drift.** The questions moved from *"is language-specific survival worse?"* (answered no) to *"does the language feature matter, and does that survive?"* (yes, it matters; preserved to INT5). They then moved to *"what do survival metrics actually measure?"* (mostly feature size and a quantizer-wide shift). Each move is documented with a reason and the earlier results are preserved. That is good practice. The cost is that almost every later study is exploratory on FLORES devtest sentences already examined. The project says this itself (`RESULTS_SUMMARY.md` header).

---

## 3. Rigor infrastructure that applies across studies

| practice | where it was used | comment |
|---|---|---|
| Freeze files committed before scoring | Belebele (`functional_validation/evaluation_freeze_mc.json`, committed 2026-10-22 15:43; statements 2026-10-23 01:17); loss task (`evaluation_freeze_lm.json`, 07:41, before the report at 09:02); Study G (design 2026-10-24 07:01; report 2026-10-25); transfer claim rules (amendment A14); survival measurement (`survival_measurement/analysis_freeze.json` 06:12, results 12:37); 9B (`d27a6a0` 01:42, verdict `0c3d3c7` 06:31, on the other branch) | Commit times checked read-only with `git log`. The order is as stated. Commit timestamps are author-set, so this is evidence of order, not proof of it |
| Identity gates | INT0 restore-only reproduces FP exactly; sham edits have harm exactly 0; "split equals full", "scale one identity", "donor self identity" gates in every Study G / 9B job; survival sidecar identity to 1e-14 | Strong |
| Reproduction controls | Clean-room recomputation of Phase 1: 89 of 89 files identical (`phase1_reproduction.md`, D-026). Old analysis code on old files byte-identical before every GPTQ analysis. Committed Spearman values reproduced before the survival-effect analysis. Stage 1 control: 486 values, largest difference 0 | Strong. Re-read here, not recomputed (needs the cached raw files) |
| Mutation testing of analysis code | Almost every module (e.g. survival_effect 16/16 mutants killed; sidecar 58/59 with one equivalent survivor; Stage 3 generator with 53 tests) | Unusually thorough for a course project |
| Multiplicity | Holm over the three Phase 1 tests; Holm in Study G extension C | **No correction** across the many exploratory families, or across the three quantizer arms (the GPTQ RESULTS says so, §6) |
| Cluster bootstrap | Language-cluster bootstrap in survival-effect and Stage 3; sentence bootstrap for group effects; crossed feature × sentence bootstrap in transfer | Group-level transfer intervals cover sentence sampling only. The groups and control draws are fixed (`intervention_transfer/results/main/claim_status.json` scope) |
| Held-out data | Study G extension C used 562 reserve devtest sentences never used before; the external Wikipedia panel (6 languages); R1 locked FLORES → Wikipedia scoring | Transfer predictive validation is **feature-held-out only**. About 88 % of held-out-feature documents also occur in calibration (`novelty_assessment_2026-10-26.md`) |
| Equivalence logic | TOST for the Phase 1 tier test (±0.02); "absent" via 90 % interval inside ±0.20 in Stage 3 H4 | Used in only two places. Most nulls are "interval includes 0" without a stated equivalence margin |
| Hardware sensitivity | D-079: an RTX 4070 vs 2080 Ti difference of about 0.0018 nats in paired harm, the same order as the Study G fine-ratio effects; the INT4 extension-C job was rerun on 2080 Ti | Important for any effect of order 1e-3. The paired arithmetic audit (D-100) was built but **never run** |

---

## 4. Study by study

Each block gives: question → design (sample sizes, controls, freeze) → result with uncertainty and source → status → rigor → limits.

### 4.1 Duan reproduction and the BOS regime (Phase 0)

- **Question.** Is our instrument valid relative to Duan (2026), who reported 51 % survival at INT6 with flat perplexity?
- **Design.** Their vendored code, unchanged except for a whitelist (D-002), on Ada (job 23472).
- **Result.** Their table was reproduced: INT7 81.19 % vs 81.26 %, INT6 51.15 % vs 51.30 % (`duan_codebase_audit.md` lines 202–203, re-read). With one BOS masked, the same WikiText windows give INT7 99.97 % and INT6 97.52 % survival, and perplexity is 16.1 instead of 458.4 (`audit_our_vs_duan.md` table, re-read). The "INT7 improves perplexity by 5.6 %" anomaly is a no-BOS artifact (A-002).
- **Status.** **ESTABLISHED.** Duan's reported survival numbers depend strongly on BOS handling. The mechanism (missing attention sink) is a **hypothesis** that was never tested (A-001).
- **Limits.** The Duan comparison was closed by user decision (D-028), so no further cross-check exists.

### 4.2 Phase 1: do language-selective features survive worse? (the "pre-set" tests)

- **Design.** 27 languages, equal token budget of 27,036 devtest tokens per language (`phase1/mask/eq/per_language.csv`). 9,500 pooled active features. Selective features were defined by S(f) above the shuffled-label 95th percentile (cutoff 1.83). 8,062 selective features against 2,843 controls matched within firing-rate strata. Survival means r ≥ 0.9. 10,000 permutations, 2,000 bootstrap draws, seed 0, Holm over three tests (D-014, D-016, D-017).
- **Results** (`phase1/mask/eq/analysis.md`, re-read):

| test | estimate [95 % CI] | p | Holm p |
|---|---|---|---|
| T1 INT7 selective − control survival | −0.28 pp [−0.63, +0.13] (99.35 vs 99.82 %) | 0.257 | 0.514 |
| T2 INT6 | −1.44 pp [−2.65, −0.13] (91.45 vs 94.20 %) | 0.043 | 0.128 |
| T3 tier slope of the gap | −0.0048 (High − Low −0.008) | 0.990 | 0.990 |

  TOST: slope 90 % interval [−0.0075, −0.0016], equivalent within ±0.02. High − Low interval [−0.0129, −0.0031], also equivalent.
- **Pooled survival** (`phase1/mask/eq/summary.csv`, VERIFIED): INT8 100 %, INT7 99.97 %, INT6 94.94 %, INT5 59.0 %, INT4 13.75 %. INT16/INT12 100 %. Perplexity change +0.08 %, +0.40 %, +2.67 %, +11.47 %, +103.9 %.
- **Exploratory follow-ups** (all post hoc, labelled so):
  - *Threshold sensitivity* (A-034): the INT6 gap is −1.02 / −2.14 / −7.16 pp at S > 2 / 3 / 5. The last two survive Holm.
  - *Magnitude matching* (A-035, D-020): matching controls on maximum activation as well turns the S > 5 gap from −7.16 pp into +0.80 (3 bins, not significant), +4.13 (5 bins) and +4.58 pp (10 bins). At the pre-set cutoff it is +1.35 / +2.69 / +3.12 pp. Selective features have lower maximum activation (median 9.8 vs 14.2 at S > 5). The JumpReLU threshold margin adds nothing, because the thresholds are nearly constant (3.00–3.19 for 80 % of features; A-040).
  - *Tier at the language level* (A-038, A-039): INT6 survival of a language's active features is 69.4 / 90.6 / 95.1 % for low / mid / high (VERIFIED). The tier slope is +12.3 pp per step (Holm 0.0002). With activation scale as covariate it is −0.81 (p 0.667). Spearman(scale, survival) is +0.966 at INT6. This is correlational and partly mechanical.
  - *Behavioural link* (A-032): Spearman(selective damage, loss increase) is +0.46 at INT6 and +0.54 at INT7 across 27 languages. It cannot be separated from overall damage (VIF up to 9.5).
- **Status.**
  - RQ1: **NULL** (small, at most about 1.5 pp at INT6; fragile under the quantizer, section 4.11).
  - RQ2 gap: **NULL with equivalence**.
  - Overall tier gap: **ESTABLISHED as an association; NOT SUPPORTED as a language-specific mechanism**.
  - "Selectivity penalty is a magnitude effect": **SUGGESTIVE → ESTABLISHED in direction**. It reverses with magnitude matching at fine bins, but the residual +3 to +5 pp is unexplained (A-035).
- **Rigor.** Clean-room reproduction 89 / 89. Holm. A TOST equivalence test. A noise-floor instrument check. The A-030 sign flip between regimes is unexplained but confined to the retired regime.
- **Limits.**
  - The "selective" class is broad: 74 % of supported features (8,086 / 10,946; A-027). The test is "most features vs a minority".
  - INT7 is at the ceiling (A-031).
  - **The pre-registration claim is not verified.** The OSF freeze status is "unverified" (V-10, Q2, D-029). Implementation estimands (stratum standardisation, equal budgets, D-013–D-016) were chosen by the analyst on 2026-10-17/18, before the run (D-017), but are not documented as part of any registration.

### 4.3 Association retention (D-035) and its follow-ups A1–A3

- **Design.** 8,086 selective features. Of these, 6,663 (82.4 %) reproduce their dev peak language on devtest in full precision ("established"). For each width: does the feature keep its original language as the unique supported peak? Run 31640; analyses reproduced byte for byte by 31919 and 31920 (`DECISIONS_FOR_REVIEW.md`).
- **Result** (`association_retention/results/eq/language_summary.csv`, VERIFIED from counts):

| | INT16 | INT8 | INT7 | INT6 | INT5 | INT4 |
|---|---|---|---|---|---|---|
| retained % | 99.2 | 98.1 | 97.5 | 94.7 | 89.1 | 70.8 |
| lost (of 6,663) | 56 | 124 | 168 | 356 | 726 | 1,945 |

- **A2** (margin, D-066): the full-precision margin predicts the loss of the tag within a language with AUROC 0.897 (INT6), 0.81 (INT5) and 0.75 (INT4) (`association_retention/results/within-language/report.md`, re-read). **A3** (D-067): the same margin predicts a *function* event weakly, AUROC 0.577 (INT6) and 0.525 (INT5), against 0.864 for the association loss on the same F5 subsample (`.../margin-function/report.md`).
- **Status.** Tag loss is mostly a margin (borderline) effect: **ESTABLISHED**. "Tag loss = function loss": **NOT SUPPORTED**.
- **Limits.** Margin and status come from the same devtest text (regression to the mean). The rounding-only floor is the only reference. The association study was **not** repeated for GPTQ or matched RTN (a 2.76 GB file per arm did not fit on the cluster disk; `gptq_extension/RESULTS.md` §6).

### 4.4 F1 / F2: does the language group matter, and does that survive quantization?

- **Design.**
  - Per language, K64 selective features zero-ablated against the mean of 3 activity-matched control draws.
  - 300 devtest sentences per language; features selected on dev.
  - Next-token loss on the language's own text.
  - 9 precisions including the INT16/INT12/INT0 controls.
  - Freeze committed before the report; analysis exploratory because devtest had been examined before.
  - Macro retention = mean of per-language ratios (D-041).
- **Result** (`functional_validation/results/eval/effects_by_language.csv`, `retention.csv`; VERIFIED):
  - FP excess harm: mean 0.7037 [0.6844, 0.7225] nats/token, median 0.2624. 27 of 27 languages have an interval above 0. The median control harm is 0.0235.
  - On the 18 Belebele-covered languages the mean is 0.713, 18 of 18.
  - Range: Bengali 0.010 to Mandarin 7.58.

| | INT16 | INT8 | INT7 | INT6 | INT5 | INT4 |
|---|---|---|---|---|---|---|
| mean of ratios | 1.00 | 1.01 | 1.03 | 1.04 [1.02, 1.05] | 1.13 [1.11, 1.18] | 2.14 [2.04, 2.50] |
| median ratio | 1.00 | 1.01 | 1.02 | 1.03 | 1.09 | 1.54 [1.43, 1.62] |

- **Status.** F1 is **ESTABLISHED** (huge and universal within the setting). F2, "kept to INT5, inflated at INT4", is **ESTABLISHED** for legacy RTN. The inflation shrinks to a median 1.22 under GPTQ (section 4.11). The rule's class becomes "inconclusive" rather than "amplified".
- **Limits.**
  - Zero-ablation shows necessity of the term, not sufficiency.
  - The controls are matched on activity but imperfectly on norm. For Llama the norms are matched at FP but not at INT4: 0.12 vs 0.19 (`RESULTS_SUMMARY.md` claim 17).
  - At INT4 the base model is already damaged (loss +0.71 nats), so a larger ablation effect is not "stronger dependence".
  - The heavy tail matters: Mandarin's 7.6 nats dominates means, hence D-041's per-language ratio rule.

### 4.5 F3: patching full-precision activations into the INT4 model (rescue)

- **Design.** At layer 12, replace the quantized activations of (a) the group's features, (b) the matched controls, (c) all SAE features, (d) the full residual with their FP values on the same tokens. The outcome is the share of the quantized-minus-FP loss gap recovered (`functional_validation/results/rescue/rescue.csv`).
- **Result** (VERIFIED, INT4, macro over 27 languages): group 2.62 % (range −0.0 to 8.7 %), controls 1.22 %, all SAE features 17.4 % (range −10.3 to 25.9 %), full residual 39.3 % (28.0 to 49.9 %). At INT5 the group recovers 0.1 % [−0.3, 0.4]. Under GPTQ: group 2.0 %, all SAE features 10.8 %, residual 41.4 % (`gptq_extension/results/stage3/analysis_gptq_rescue/rescue.csv`, VERIFIED).
- **Status.** The measurement is **ESTABLISHED**: the specified group patch repairs little. The interpretation "the damage does not sit in the language features" is **NOT SUPPORTED as a causal claim**. Repair shares of different patches are neither additive nor upper bounds on causal responsibility (correction D-110, `PROJECT_RESULTS.md` §4.5). Partial patches can make the loss worse, which shows the components interact.
- **Limits.** One layer. One donor/patch construction. Nothing was measured above layer 12.

### 4.6 F4: does the association status predict function at equal activity?

- **Design.** At INT6 / INT5 / INT4, features that lost their tag vs activity-matched features that kept it. Group harm compared (`functional_validation/results/f4/`). INT4: 26 languages (por_Latn not in the analysis); INT5: 24; INT6: 18.
- **Result.** Median lost − retained difference in group harm: INT4 +0.0006 [−0.0001, +0.0012], INT5 +0.0007, INT6 +0.0002 nats/token. The INT4 macro mean is carried by Lao. *Auditor's note:* without Lao the INT4 mean is still +0.0032 [+0.0026, +0.0038] (`results/f4/INT4/report.md`, post hoc robust summary). The FP mean on the same subsets is already +0.0014 [+0.0010, +0.0019]. The "no systematic difference" reading rests on the median and on Delta(INT4 − FP) = −0.0005 [−0.0012, +0.0001].
- **Status.** **NULL** for a meaningful difference: effects of a thousandth of a nat, and present at FP already. The INT4 mean without Lao is formally non-zero, so the wording "no systematic difference" is slightly too strong (section 6.2).
- **Limits.** Feature sets come from the legacy quantizer's association losses, so the analysis was not repeated for GPTQ.

### 4.7 F5: does a single feature do the same thing after quantization?

- **Design.**
  - 1,487 features from all 27 languages, a stratified (not random) sample (D-056).
  - The effect fingerprint is the change of output logits when the single feature is switched off, over M = 16 contexts.
  - W1 is the cosine between the FP and quantized fingerprints.
  - The numerical floor is the 5th percentile of INT16, 0.949 (D-055, after A-042 corrected the wrong assumption that INT16 equals FP).
  - Controls: a swapped direction (another feature's decoder direction at the same sites and amplitudes), fixed-amplitude variants, and M = 8 / 32 sensitivity (D-060–D-064).
  - The rules S1–S3 were fixed before the first number (D-057).
- **Result** (`functional_validation/results/f5/report.md`, re-read from the generated report):

| state | median W1 [95 %] | share ≥ floor | median R1 | R1 ≥ 0.9 | identity retrieval |
|---|---|---|---|---|---|
| INT8 | 0.997 [0.996, 0.997] | 0.940 | 0.990 | 0.999 | 1.000 |
| INT6 | 0.988 [0.986, 0.989] | 0.897 | 0.961 | 0.898 | 1.000 |
| INT5 | 0.961 [0.958, 0.964] | 0.665 | 0.909 | 0.560 | 1.000 |
| INT4 | 0.839 [0.829, 0.848] | 0.003 | 0.738 | 0.132 | 0.997 |

  - Swapped-direction control at INT4: 0.828 against 0.842 for the features. With FP amplitudes everywhere: 0.838 against 0.846.
  - At INT8 / INT6 / INT5 the control is only 0.004 / 0.006 / 0.007 below the features (D-061, D-063, D-064).
  - About 31 % of functionally intact features at INT5 are counted as "lost" by the survival rule.
  - S1 met (0.902, uninformative: 87 % of "lost" features also pass). S3 met (0.824). S2 undefined (2 events).
- **Under GPTQ** (`gptq_extension/results/f5/*/report.md`, re-read): median W1 at INT4 is 0.839 / 0.875 / 0.923 and the share at the floor is 0.3 / 0.7 / 11.7 % (legacy / matched / GPTQ). The S2 AUROC under GPTQ is 0.490 [0.431, 0.560] with 79 events: **not met**.
- **Status.**
  - "Features remain identifiable and their effects are near-preserved to INT6": **ESTABLISHED**.
  - "The drift is mostly generic, not feature-specific": **ESTABLISHED** in direction. The feature-specific margin of 0.003–0.014 has **no intervals**, so it is **SUGGESTIVE** at most.
  - "Association status predicts function": **NOT SUPPORTED** (AUROC 0.49–0.51).
- **Limits.**
  - Stratified sample.
  - One matched control per feature in the swap control (404 features).
  - Because the floor is the 5th percentile of INT16, the share "within noise" cannot exceed about 0.95 by construction.

### 4.8 Belebele: the only confirmatory downstream family

- **Design.**
  - Freeze `evaluation_freeze_mc.json` committed before scoring.
  - 18 of 27 languages are "covered": the model's accuracy is clearly above chance. 385 devtest-origin items each. Zero-shot letter format.
  - K64 group vs 3 matched control draws, zero-ablation.
  - S0 (FP, one-sided 95 %): the macro effect E > 0. S1–S5 (retention by precision) are conditional on S0 (D-044, D-047, D-050).
- **Result** (`functional_validation/results/mc_eval/`, VERIFIED):
  - Accuracy at FP: none 0.4257, group zeroed 0.4255, controls 0.4251.
  - S0: macro E −0.0001 [−0.0035, +0.0031]. The unit is nats per item (log-probability of the correct option), not accuracy. 10 of 18 languages have E > 0. **Not met.** S1–S5 are undefined.
  - Accuracy without edit: INT8 0.426, INT7 0.420, INT6 0.423, INT5 0.387, INT4 0.236 (below the 0.25 chance level).
- **Sensitivity checks (exploratory).**
  - K = all (up to 1,719 features): 0.4257 → 0.4209, controls 0.4226 (VERIFIED; D-062).
  - Positive control (D-069, D-072; `results/mc_posctrl/summary.md`, re-read):
    - random non-selective sets lower accuracy by −0.76 [−1.24, −0.26] (256 features), −1.20 (1,024) and −2.19 (all 2,860);
    - control sets drawn from other languages' selective features lower it by up to −6.01 points;
    - the sham changes it by exactly 0.
    - The pre-written P1 (≥ 5 points from the whole non-selective pool) was **not met**, and P4 was **reversed**.
  - Prompt check (D-086): the frozen prompt beats the lm-eval-harness prompt in 18 of 18 languages (difference −0.100 [−0.110, −0.090]).
  - GPTQ INT4 model (accuracy 0.318): zeroing the group changes accuracy by −0.29 points vs no edit and −0.44 vs controls (`gptq_extension/results/stage3/analysis_gptq_mc/accuracy.csv`, VERIFIED).
- **Auditor's approximate bound** (my computation, not the project's): a language-level bootstrap of the accuracy difference "group zeroed − controls" over the 18 languages gives +0.04 pp [−0.43, +0.57] for K64 and −0.17 pp [−0.95, +0.77] for K = all (`/tmp/opus_analysis/`, from `accuracy.csv`). This supports the project's informal "at most about half a point" for K64. For K = all the bound is closer to one point.
- **Unreported anomalies in the context table** (`results/mc_eval/statements.md`): macro E at INT5 is −0.0066 [−0.0102, −0.0031] and at INT4 +0.0042 [+0.0003, +0.0081]. Both intervals exclude zero, in opposite directions. The INT4 model is at chance and these are exploratory, so they are probably noise or artefacts. They are nonetheless not mentioned in `PROJECT_RESULTS.md`.
- **Status.** **NULL** (confirmatory), with demonstrated sensitivity to edits of 1 to 6 points. Why loss and comprehension dissociate: **NOT TESTED**. Three hypotheses are listed in `PROJECT_RESULTS.md` §1 and none was tested.
- **Limits.** 18 languages, 43 % baseline accuracy (a blunt instrument), one layer, zero-ablation only. Absolute accuracies depend on the prompt by about 10 points.

### 4.9 Study G: graded gain (scale the group down and up), extensions B, C, D

- **Design.**
  - 27 languages × 6 states (FP, INT8–INT4) × 300 sentences × 3 control draws. This is the user's sample-size floor (D-075).
  - Ratios: down 0.354 / 0.177 / 0.088 / 0; up 2.83 / 5.66 / 11.3; fine 1.1 / 1.2. Calibrated on 100 FP pilot sentences (D-071).
  - Rules H-G1–H-G7 fixed in the design before the data (spec committed 2026-10-24 07:01).
  - Exploratory, because devtest had been used before.
- **Results** (`functional_validation/gain_study/results/`, VERIFIED unless noted):
  - H-G1 dose, down: FP median net harm 0.0223 / 0.0902 / 0.1687 / 0.2624 at 0.354 / 0.177 / 0.088 / 0. 27 of 27 strictly ordered. **met**
  - H-G2 down is specific: 26 of 27 have a lower bound > 0 at 0.088. **met**
  - H-G3 dose, up: 27 increasing and 24 net-positive at 11.3. **met**
  - H-G6 consistency: 27 of 27 in every state; Spearman FP vs INT4 0.936. **met**
  - H-G4 precision: **narrowly not met**. The median falls by more than one bootstrap error from FP to INT8. It would be met with two errors, a post hoc sensitivity.
  - H-G5a/b repair: **not met**, under-powered (12 of 27 against 14 required).
  - H-G5 α* vs 1/g: −0.034, **not met**.
  - H-G7 baseline: **not met** (mass share does not fall).
  - Zero-ablation net harm by state (median): FP 0.262, INT8 0.268, INT7 0.266, INT6 0.271, INT5 0.281, **INT4 0.432**. Flat to INT5, then a step.
  - At ×0.088: INT4 0.323 vs FP 0.169, a factor of 1.92. At ×11.3: FP 1.213, INT4 0.810.
  - Fine ratios at INT4: ×1.1 net −0.0010 (21 of 27 negative), ×1.2 −0.0013 (18 of 27). At FP: +0.0008 and +0.0020.
  - Best gain α* > 1 at INT4 in 21 of 27 languages, against 7 of 27 at FP and INT6 (`alpha_star.csv`).
- **Extension C** (D-082; 562 reserve sentences, 5 control draws, Holm; `extension_c/cells.csv`, VERIFIED):
  - INT4 ×1.1: −0.00135 [−0.0020, −0.0003], 20 of 27 negative (sign p 0.0096). **met**
  - INT4 ×1.2: −0.0017 [−0.0031, +0.0003], 18 of 27. **not met**
  - INT4 − FP: −0.0024 [−0.0042, −0.0012] (×1.1) and −0.0041 [−0.0071, −0.0025] (×1.2).
  - Controls −0.0002 [−0.0004, 0.0000], so group-specific "by the rule, narrowly".
  - Shape: minimum near 1.1–1.2, gone by 1.4 (−0.0007), a cost at 1.7 (+0.0046).
  - The first INT4 job ran on an RTX 4070 and was rerun on 2080 Ti because of a hardware shift of similar size (D-079).
- **Extension D** (D-083/D-084; post hoc, rules committed before the module and the run): α* vs inverse shrinkage, Spearman −0.170, one-sided p 0.80. Split-half reliability of α* is 0.846 (Spearman-Brown 0.917). Placebo +0.186. **NOT SUPPORTED** (`extension_d/analysis.json`, VERIFIED).
- **Extension B** (D-081/D-085): groups re-selected in the quantized model overlap the FP group with Jaccard 0.855 (INT6) and 0.488 (INT4), 27 of 27. They are not more consequential: zero-ablation at INT4 is 0.452 (state-defined) vs 0.477 (FP reference), positive in 11 of 27. H-Gb2 **not met** (`state_defined/results/report.md`, re-read).
- **Under GPTQ** (D-109, VERIFIED):
  - INT4 ×0.088: 0.323 / 0.281 / 0.187 (legacy / matched / GPTQ), so the amplification shrinks from 1.9× to 1.1× FP.
  - α* > 1 at INT4: 21 / 13 / 7 of 27. GPTQ is the same as FP.
  - ×11.3 at INT4: 0.810 / 0.854 / 1.146 (FP 1.213).
- **Status.**
  - Graded, specific, consistent dose-response of the group: **ESTABLISHED**.
  - "A threshold at INT4, not a gradual slide": **ESTABLISHED for legacy RTN**, much weaker under GPTQ. This is a property of the quantizer's damage level, not a universal threshold.
  - INT4 fine-gain "repair": **SUGGESTIVE for legacy RTN only**. It replicates on fresh sentences, but the size is about 0.04 % of the loss and of the order of the hardware shift. It **disappears under GPTQ** (7 of 27 = FP), so it is **NOT SUPPORTED** as a general phenomenon.
  - Shrinkage mechanism: **NOT SUPPORTED**.
  - Choice of the FP group does not drive the INT4 results: **ESTABLISHED (null for the median)**.

### 4.10 Llama-3.2-1B consistency check (D-087, D-090)

- **Design.**
  - Llama-3.2-1B; `seonglae/Llama-3.2-1B-sae` FineWeb TopK (k = 48, 14,336 features) at `blocks.12.hook_resid_pre`, layer 12 of 16.
  - Same languages, 300 sentences, 3 draws, states FP / INT8 / INT6 / INT5 / INT4.
  - Predictions P1–P3b and the verdict function were committed before any Llama number. The comparison module was committed 03:56:50, before data.
- **Result** (`functional_validation/llama/analysis.json`, `llama_robust/robust.json`, VERIFIED):
  - P1: median E +0.048 [0.018, 0.173] against Gemma's 0.262. 21 of 27 languages > 0, with 24 required: **not met**. With the median of the control draws it is 23 of 27 (post hoc).
  - P2a: retention INT8 1.010, INT6 1.024. **met**
  - P2b: frozen mean retention at INT4 −3.57 (dominated by a language with a tiny FP effect). Post hoc median 0.856 [0.743, 1.029] over 21 languages. **not met**
  - P3a: ordered in 20 of 27 at FP and 15 of 27 at INT4. **not met**
  - P3b: 14 of 27 vs 20 required, median 1.17. **not met**
  - Verdict: **"not consistent"**.
- **Status.** Cross-model generalisation of the Gemma picture: **NOT SUPPORTED** on this pair. What caused the difference: **INCONCLUSIVE**. Model family, SAE type and recipe, depth and tokenizer all differ. This SAE labels 91 % of features language-selective (Gemma about 75 %). The control pool is 898 vs 2,860.
- **Limits.** Not a replication; three control draws, heavy-tailed.

### 4.11 GPTQ / matched-RTN extension (D-101–D-111)

- **Design.**
  - Same model, SAE, sentences, groups, controls, batching and analysis code. Only the weights differ.
  - Matched RTN separates the "implementation of RTN" from GPTQ's compensation.
  - GPTQ calibration: 221,184 Wikipedia tokens, disjoint from FLORES and from the external panel.
  - Gates:
    - weight-set sha256 equal across more than 10 nodes;
    - FP-dependent numbers bit-identical to the old runs (Phase 1 exact; transfer 259,000 items, 0 differences; Belebele FP 0.4257);
    - old analysis code reproduces the old outputs exactly before every new analysis.
  - Exploratory. No multiplicity rule covers three arms.
- **Results** (VERIFIED from `gptq_extension/results/*`):

| quantity | legacy RTN | matched RTN | GPTQ | source |
|---|---|---|---|---|
| survival INT6 (mean of 27 languages) | 85.0 % | 89.7 % | 97.5 % | `stage1/stage1_paired_per_language.csv` |
| survival INT4 | 12.7 % | 18.6 % (corrected from 18.7) | 30.5 % | same |
| loss increase INT6 / INT4 (nats/token) | 0.0254 / 0.590 | 0.0197 / 0.385 | 0.0058 / 0.130 | same |
| GPTQ − legacy survival gain at INT6 by tier (low / mid / high) | | | +25.2 / +8.0 / +4.3 pp | same |
| Phase 1 T1 / T2 Holm p | 0.514 / 0.128 | **0.042 / 0.038** | 1 / 1 | `stage1/analysis_*/analysis.md` |
| transfer normalized error INT4 / INT6 | 0.664 / 0.165 | 0.572 / 0.153 | 0.476 / 0.132 | `stage2/transfer_feature_level.csv` |
| error over floor INT6 | 1.88 | 1.75 | 1.52 | same |
| group ablation median ratio INT4 | 1.54 | 1.36 | 1.22 | `stage3/analysis_*_eval/retention.csv` |
| mean E at INT4 | 0.935 | 0.900 | 0.779 | same |
| Belebele accuracy INT4 (no edit) | 0.236 | 0.291 | 0.318 (corrected from 0.319) | `stage3/analysis_*_mc/accuracy.csv` |
| external panel cosine INT4 | 0.754 | 0.808 | 0.876 | `external/transfer_feature_level.csv` |
| transfer UTILITY, main / external | not met / met | not met / met | not met / **not met** | `*/transfer_claims.csv` |

  GPTQ beats legacy RTN in 27 of 27 languages on survival and loss at INT4 and INT6. At INT4 GPTQ reduces the group's total transfer harm by 0.46 [0.31, 0.62] nats/token, in 26 of 27 languages (`stage2/transfer_group_summary.csv`, VERIFIED).
- **Status.**
  - "Effect magnitudes depend strongly on the quantizer; directions mostly do not": **ESTABLISHED**.
  - "The 6-bit low-resource survival gap is mostly a property of the legacy quantizer": **ESTABLISHED descriptively** (+25 pp gain for the low tier).
  - "Phase 1 null is robust": **NOT SUPPORTED**. Under matched RTN two tests cross Holm 0.05. With a factor 2 for the extra families they would not (0.084 / 0.077). The effect is ≤ 1.5 pp. Call it **INCONCLUSIVE / fragile**.
  - "GPTQ is better because of error compensation": **NOT TESTED**. Calibration and compensation are confounded, and the calibration domain is Wikipedia, which matches the external panel.
- **Limits.** INT6 and INT4 only, except stage 1. F4 and association retention not repeated. Belebele S1–S5 not rerun.

### 4.12 Intervention transfer: context vs weights (D-088, D-094, D-095) and recovery R1–R4 (D-096–D-099)

- **Design.**
  - A fixed FP feature-direction edit, applied in a crossed 2 × 2: quantized or FP layers below the edit (context C) × quantized or FP layers above it (weights W), plus interaction I.
  - Group level: K64 groups, 27 languages, 300 sentences, 3 controls.
  - Feature level: 404 features, 6,464 real-direction items. 2,203 items (235 features, 34.1 %) are "eligible", meaning their FP effect clears the numerical floor.
  - Controls: matched swapped direction and 3 random directions.
  - Held-out ridge predictor of transfer error (feature-held-out halves, 1,066 held-out items).
  - Claim rules fixed before outputs (amendment A14). Crossed feature × sentence bootstrap, 2,000 replicates.
  - External panel: Wikipedia, 6 languages (eng, hin, rus, spa, yor, zho).
- **Group-level results** (main, `intervention_transfer/results/main/group_per_language.csv`, VERIFIED medians over 27 languages):

| state | FP net harm | both quantized, fixed edit | own (native) activation | weights only | \|C\| | \|W\| | \|I\| | C > W |
|---|---|---|---|---|---|---|---|---|
| INT16 | 0.263 | | | | 0.0005 | 0.0003 | | **19 / 27** |
| INT6 | 0.263 | 0.281 | 0.270 | 0.272 | 0.0091 | 0.0092 | 0.0010 | 12 / 27 |
| INT4 | 0.263 | 0.848 | 0.433 | 0.428 | 0.310 | 0.104 | 0.071 | 19 / 27 |

  A3 ("context dominates") is supported at INT4, inconclusive at INT6, **mixed overall**. Note that **C > W in 19 of 27 also occurs at INT16**, where nothing is quantized meaningfully. The count is therefore not evidence; only the interval of the median difference is (D-094 update of 21:47). With the native activation, the INT4 harm (0.433) equals the weights-only cell (0.428). The large context term belongs to the *fixed* edit's amplitude mismatch.
- **Feature-level results** (`main/analysis_summary.json`, n = 2,203 eligible items, VERIFIED):
  - INT6: cosine 0.984, normalized error 0.165, about 1.9 floors.
  - INT4: cosine 0.768, normalized error 0.664, norm ratio 1.09. "Rotated, not shrunk."
  - The weights term has the larger norm (relative 0.560 vs 0.359 at INT4). This is opposite to the group-level loss result, because the two estimate different things.
  - Real vs control normalized-error gap +0.019 [+0.009, +0.032] at INT4. Mostly generic.
  - Averaged fingerprints overstate transfer: +0.126 cosine at INT4 vs +0.015 at INT16 (H4).
- **Predictive validation** (`main/predictive_validation.csv`, VERIFIED):

| state | A+B over strongest baseline [95 %] | criterion (≥ 10 %, lower bound > 0) |
|---|---|---|
| INT16 | 23.4 % [15.1, 30.8] | met (rounding-only state) |
| INT6 | 10.9 % [−1.2, 21.4] | not met |
| INT4 | 21.5 % [12.7, 29.4] | met |

  UTILITY overall: **not met** (main). The external panel was **met** (INT4 19.4 %, INT6 23.4 %).
  *Auditor's note:* at INT6 the ridge baseline A has a larger held-out MAE (0.0290) than the training-median null (0.0271). Against that null, A+B improves by only about 4.5 %. This is what R1 later formalised.
- **Recovery analyses** (post hoc; protocols hashed before the code; `results/recovery/`, VERIFIED):
  - **R1, locked FLORES → Wikipedia** (1,368 eligible items): INT16 28.2 % [21.3, 35.1]; INT6 18.6 % [4.9, 29.5], but **15.8 % [−2.7, 28.7]** against a median null; INT4 19.9 % [7.8, 31.3].
  - **R2, gradient scalars vs real loss change:** r = 0.63 [0.46, 0.80] on FLORES.
  - **R3, support-matched real vs swapped:** error ratio ×1.12 (INT6) and ×1.14 (INT4), log2 0.165 and 0.183. Only 144 FLORES pairs (about 2 %).
  - **R4, native vs fixed:** at INT4 the native harm is below the fixed harm in 27 of 27 languages (median −0.351 [−0.358, −0.293]). The residual over weights-only is +0.014 in the median language, ranging −0.78 to +0.23.
- **External group level.** `results/external/README.md` reports INT4 |C| 0.133, |W| 0.073 and "INT6 reversed". These use the analysis code's **lower median** (`scripts/g0/phase1_transfer_analysis.py`, `weighted_median`). With six languages the conventional median gives |C| 0.333, |W| 0.096 at INT4 (`PROJECT_RESULTS.md` table A16). At INT6 it gives 0.0185 vs 0.0134, the opposite order to the reported 0.0076 vs 0.0124. The paired median difference is negative under both conventions, and C > W holds in only 2 of 6 languages (section 6.5, item 36).
- **Status.**
  - "Effect near-preserved at INT6, rotated at INT4, largely generic": **ESTABLISHED**.
  - Direction-specific component (×1.12–1.14): **SUGGESTIVE** (2 % of pairs, post hoc).
  - "Context vs weights": **INCONCLUSIVE**. It depends on the estimand (loss vs vector norm), on fixed vs native edits, on the corpus and, for n = 6, on the median convention.
  - Generic-sensitivity predictor: **ESTABLISHED as an association**. It is **NOT SUPPORTED as quantization-specific**: it predicts INT16 equally well. **NOT SUPPORTED as a cheap audit**: it needs four control interventions per context.
  - **Document leakage:** about 88 % of held-out-feature documents occur in calibration. On 136 items from 65 unseen documents the gains are 6.7 % (INT4), 0.5 % (INT6) and 29.4 % (INT16), without intervals (`novelty_assessment_2026-10-26.md`; `publication_reassessment_2026-10-26/README.md`).
- **Not run:** the paired arithmetic-precision audit R5 (D-100), built and tested with 476 tests but never deployed.

### 4.13 Survival versus effect (D-110–D-114; post hoc, frozen before running)

- **Design.**
  - Unit: the F5 feature, 1,412 "reliable" of 1,487.
  - R1 (activation correlation) as predictor of W1 (effect similarity), beyond baselines A (activity, firing rate, mean activation, |dNLL_FP|, ablated positions).
  - Leave-one-language-out ridge. Language-cluster bootstrap, 2,000 replicates.
  - Rule: B over A ≥ 10 % with lower bound > 0.
  - 19 tests, 16 / 16 mutants killed. Committed ρ values reproduced first.
- **Result** (`survival_effect_rescue/results/survival_effect_results.json`, VERIFIED):
  - Spearman R1–W1: 0.30–0.52. Partial Spearman given the baselines: 0.24–0.40.
  - B over A: legacy INT6 4.4 % [2.6, 6.2], INT4 3.7 % [2.3, 5.2]; matched 3.8 % / 3.1 %; GPTQ 4.1 % / 3.5 %. 0 of 6 cells meet the rule.
  - The baselines themselves beat the median null by only 0.2–3.2 %.
  - In the noisy-effect stratum (n = 75) two cells exceed 10 % (15.6 % and 15.0 %). This is secondary, and `PROJECT_RESULTS.md` does not report it.
  - Llama-3.2-1B (643 features, 610 reliable): INT6 −0.7 % [−2.0, 0.4], with the association not resolvable; INT4 4.2 % [1.6, 6.7], partial 0.252.
- **Status.** A weak positive association: **ESTABLISHED**. Useful predictive value beyond baselines: **NULL** against the pre-set 10 % convention. The improvement is real but small: 3–4 % with intervals above 0. The baselines are almost useless too, so the outcome W1 is hard to predict at all.
- **Limits.** R1 and W1 come from the same perturbed model. Held out by language, not by document. Stratified sample.

### 4.14 Survival measurement-validity study (D-117–D-124; Stage 3 frozen before data; post hoc relative to Stage 1)

- **Design.**
  - 214,506 FP-active (language, feature) cells; 27,036 tokens per language.
  - Three quantizers at INT6/INT4, plus legacy INT16.
  - A GPU sidecar streamed the sufficient statistics:
    - gain g;
    - residual noise σ_e;
    - λ = log noise-to-signal;
    - on/off flips;
    - ν = encoder pre-activation perturbation in units of the feature's threshold.
  - Gates, all held: reproduction of committed survival to 3e-8; FP side bitwise identical across jobs; identity r(g, σ) = r(sums) to 1e-14.
  - D-121 (user): thresholds are conventions, and every improvement is reported with its interval.
  - D-123 changed the generator after the freeze but before any Stage 3 number, all documented: an FP-drift tolerance, the selectivity comparison group and wording.
- **Results** (`survival_measurement/stage3/stage3_results.md` / `.json`, VERIFIED for g and ν-by-tier):

| | legacy INT16 | legacy INT6 / matched / GPTQ | legacy INT4 / matched / GPTQ |
|---|---|---|---|
| median gain g | 0.997 | 0.958 / 0.967 / 0.976 | **0.594 / 0.672 / 0.823** |
| active tokens switched off | 1.1 % | 9.2 / 7.7 / 5.9 % | **50.0 / 42.9 / 27.0 %** |
| median ν | 0.018 | 0.130 / 0.113 / 0.086 | 0.706 / 0.562 / 0.378 |

  - **H3:** λ correlates with the log feature spread at −0.507 to −0.534 (convention |ρ| > 0.5 only just met) and with the log peak at −0.72 to −0.75. ν is nearly free of the spread (|ρ| ≤ 0.114) but not of the peak (0.25–0.41).
  - **H4:** median ν by tier (low / mid / high) at INT16 is 0.0192 / 0.0179 / 0.0170 (VERIFIED), already 13 % higher for low. The tier slope is 0.063 at INT16 and 0.077–0.085 at INT6/INT4 (RTN), so the quantization-specific part is absent or inconclusive. GPTQ is "present" (s = −0.79, −0.90), but it reverses to inconclusive with a spread covariate.
  - **H5:** g and ν on top of R1 add −0.4 % to +1.8 %, every interval including 0.
  - Selectivity (descriptive): selective cells have ν 14–19 % lower.
- **Stage 1** (`survival_measurement/stage1/README.md`, re-read):
  - The same features are fragile under every quantizer (median Spearman of λ 0.91–0.96).
  - One global λ shift reproduces matched-RTN survival rates within 0.6 / 1.4 pp, but not GPTQ at INT4 (4.7 pp).
  - The tier slope of −median λ is 0.42 at INT16 and 0.43–0.49 at INT7–INT5. With the scale covariate it is 0.09–0.18, still p ≤ 0.017. That contradicts the Phase 1 reading that "scale carries the whole tier effect" for the threshold-free measure.
- **Status.**
  - "Survival rates mix quantizer and feature size; a fixed cutoff magnifies a global shift": **ESTABLISHED**.
  - "ν removes size": **partly** (spread yes, peak no).
  - "Tier pattern is quantization-specific": **NOT SUPPORTED** under RTN. **INCONCLUSIVE** under GPTQ.
  - "Gain and ν improve function prediction": **NULL**.
- **Limits.** One model, layer and SAE. About 23 effective languages for tier slopes. Several H4 labels depend on the standardisation.

### 4.15 Gemma-2-9B, original RTN, imported (D-125) — UNVERIFIED

- **Design** (from the imported freeze and verdict):
  - `google/gemma-2-9b`, Gemma Scope `layer_20/width_16k/canonical`, `blocks.20.hook_resid_post`.
  - The same 300 sentence ids and FLORES corpus as 2B (task 1 of the review confirmed hashes).
  - 6 states, 3 control draws, K64.
  - Study G ratios taken from 2B, not recalibrated.
  - Selective features: 5,218 of 16,384 (31.8 %) vs 8,086 (49.4 %) on 2B.
  - One RTX PRO 6000. All four gates true.
- **Reported** (`functional_validation/gemma9b/verdict.md`): P1, P2a (INT8 1.003, INT6 1.006), P2b (INT4 2.479), P3a and P3b all met, so "Consistent". H-G5b repair 8 of 27 (14 needed).
- **Re-computed from the imported `results/analysis/curves.csv`** (a derived table, not raw data):
  - FP zero-ablation net harm is positive with a lower bound > 0 in 27 of 27 languages. Median 0.270, mean 0.794.
  - Mean-of-ratios retention: INT8 1.003, INT7 1.047, INT6 1.006, INT5 1.205, INT4 2.479. This matches the verdict.
  - **Median** ratio at INT4 is 1.586, which the verdict does not report. The mean is driven by Kannada (9.9×, FP effect 0.010) and Korean (7.6×).
  - Harm increases strictly as the ratio falls (0.354 < 0.177 < 0.088 < 0) in 27 of 27 languages at FP.
  - In the Llama-P3b sense (INT4 / FP net harm at ×0.088): 26 of 27 larger, median 1.69.
  - INT4 ×1.1: median net −0.0005, 17 of 27 negative. The same sign as 2B legacy, but smaller.
- **Open problems** (`gemma9b/REVIEW_STATUS.md`, plus this audit):
  1. Raw per-state files have not been fetched or reproduced.
  2. `meta_lm_eval.json` lists only INT4.
  3. **P3a as frozen is literally reversed.** It says "E(0.354) > E(0.177) > E(0.088)", but harm grows as the ratio falls. Read literally, 0 of 27 languages meet it. The verdict silently uses the intended direction.
  4. **P3b changed meaning.** In the Llama plan, P3b is the INT4-over-FP ratio at the deepest down ratio. The 9B freeze says "median selectivity ratio above 1.3" and the verdict describes it as "features are strongly language-selective". In the Llama sense it would also be met (26 of 27, median 1.69).
  5. The P1 confidence interval is required by the freeze but not reported.
  6. The verdict's "consistent with the 2B observation that quantization damage concentrates on language-selective circuits" contradicts the 2B rescue result (2.6 %). It is already caveated by the consolidation note.
- **Status.** **UNVERIFIED / SUGGESTIVE**. The imported derived tables are internally consistent with the verdict on P1, P2a, P2b and the substance of P3a. Independent raw reproduction is pending. No GPTQ or matched-RTN 9B arm exists.

### 4.16 Not tested (planned or proposed, never run)

| item | status | where recorded |
|---|---|---|
| Paired arithmetic-precision audit (R5) | built, 476 tests, never deployed | D-100, Q29 |
| Real integer kernels (e.g. GPTQ / AWQ INT4 kernels), activation quantization | not studied | `PROJECT_RESULTS.md` §4.12, §5 |
| Other layers, other SAEs on the same model | not studied | same |
| GPU diagnostic of why low-resource languages have a lower activation scale (Q15) | job never designed or run | `INDEX.md` §3 |
| Random-weight-noise controls and all-token intervals (Q16 options D, E) | not selected | D-039 |
| Full 4,382-feature F5 sample (Q24) | default no | `DECISIONS_FOR_REVIEW.md` |
| GPTQ without calibration, or with a different calibration set | not run (it would be a new experiment) | `gptq_extension/RESULTS.md` §6 |
| 9B GPTQ / matched-RTN arms and 9B survival-measurement study | not run | D-125 |
| Steering / cross-language patching repair pilot | assessed as feasible but not adopted | `steering_feasibility_review_2026-10-28.md` |

---

## 5. Cross-study synthesis

### 5.1 Where the studies agree

| finding | supporting studies | strength |
|---|---|---|
| Survival / association metrics are weak proxies for functional continuity | F4 (Δ about 0.0006), F5 (status AUROC 0.506; 31 % misclassified at INT5), A3 (margin AUROC 0.58 for function), survival-effect (3–4 % gain; Llama the same), Stage 3 (g, ν add nothing), GPTQ F5 (S2 AUROC 0.49) | **ESTABLISHED** (six independent analyses, two models for one of them) |
| What survival rates mostly reflect is feature size and a quantizer-wide shift | Phase 1 magnitude matching (A-035), A-039 (scale ρ 0.97), Stage 1 (one λ shift), Stage 3 H3 (λ ~ spread −0.5, peak −0.7) | **ESTABLISHED** |
| The low-resource survival gap is not a language-specific quantization effect | A-039 (scale covariate removes it), GPTQ (+25 pp for the low tier at INT6), Stage 1 / Stage 3 (present at INT16) | **ESTABLISHED** for "not quantization-specific". *Why* low-resource features have smaller scale and higher perturbation sensitivity: **NOT TESTED** |
| Language groups matter for own-language next-token loss | F1, Study G (H-G1/H-G2), transfer FP cells, GPTQ stage 3 (27 / 27 in every arm), 9B (imported) | **ESTABLISHED** (Gemma) |
| The effect of the group is preserved through INT6 (and INT5) | F2, Study G, transfer (cosine 0.984), F5, all three arms | **ESTABLISHED** |
| At INT4, changes are large and mostly generic | F5 swap control, transfer specificity (+0.019 gap), F3 controls (1.2 % vs 2.6 %), rescue all-SAE vs group | **ESTABLISHED** |
| Quantizer quality scales all INT4 effects | GPTQ extension across every study; Stage 3 (gain 0.59 → 0.82) | **ESTABLISHED** |

### 5.2 Where the studies conflict or need reconciliation

| tension | details | resolution status |
|---|---|---|
| Next-token loss vs Belebele | The same edit raises loss in 27 / 27 languages but leaves accuracy unchanged within about ±0.5 pp | **Unexplained**; three untested hypotheses |
| "Tier effect explained by scale" vs "tier effect persists" | Phase 1 A-039: the slope goes to −0.81 with the scale covariate (thresholded survival). Stage 1: the λ tier slope stays 0.09–0.18 (p ≤ 0.017) with the same covariate and is present at INT16 | Different outcome measures. The thresholded survival rate hides a residual that the threshold-free λ shows. `PROJECT_RESULTS.md` §1 still says "shrinks to nothing" without this nuance |
| Context vs weights | Group-level fixed edit: C ≫ W at INT4. Native edit: harm ≈ weights-only. Feature vector: W norm > C norm. External INT6: W > C only under the lower-median convention | **INCONCLUSIVE**; the answer depends on the estimand |
| INT4 "repair" by turning the group up | Legacy RTN: replicated (extension C). GPTQ: gone (7 / 27 = FP). 9B legacy: weaker (17 / 27 negative, median −0.0005) | Most likely an artefact of legacy-RTN shrinkage (gain 0.59). Yet extension D found no link to the group's own shrinkage. **Unresolved, but moot** given its size |
| Phase 1 null | Legacy: Holm 0.13. Matched RTN: 0.038. GPTQ: 1 | Small effect near the ceiling; significance verdict fragile. **INCONCLUSIVE** for "≤ 1.5 pp effect exists" |
| Gemma vs Llama vs 9B | 2B: amplification 1.54 (median). Llama: 0.86 (no amplification). 9B: 1.59 median / 2.48 mean (unverified) | Same-family 9B appears consistent; a different family and SAE does not. Model family vs SAE type **cannot be separated** with these data |

### 5.3 What the whole body of evidence supports about the central question

**Central question:** does compressing the model damage the language-specific features of the SAE, and does that hurt the language?

- **Survival of language features.** Under weight-only RTN or GPTQ to INT6, language-selective features do **not** survive worse than matched features in any way that is robust and large. They actually perturb less in threshold units (Stage 3 selectivity). The apparent low-resource disadvantage is a scale or perturbation-sensitivity property that already exists at INT16. It is strongly reduced by a better quantizer.
- **Function of language features.** The groups are causally necessary for own-language next-token prediction in this model (an intervention effect, ESTABLISHED). Their effect is preserved through INT6. At INT4 the measured ablation harm is inflated in a quantizer-dependent way.
- **Does compression hurt the language through these features?** **Not shown.**
  - The group patch repairs about 2–3 % of the INT4 loss gap, while the residual repairs about 40 %.
  - Ablation-harm amplification at INT4 does not locate the quantization damage in the language features.
  - Most INT4 drift is shared with arbitrary directions.
  - On the reading task the features do not matter measurably at all.

### 5.4 Causal claims the data do NOT support (explicitly)

1. "Quantization damage concentrates in language-selective features or circuits." This was stated in the 9B verdict and earlier in `RESULTS_SUMMARY.md` claim 4 ("at most about 3 % … sits"). Amplified ablation harm is a change in the *effect of an edit* in an already-perturbed model. It does not identify where the *quantization* effect is mediated. Patch-repair shares are not causal shares or upper bounds.
2. "Low-resource languages are hurt *because* their language features break." Tier is an observational attribute. The gap is explained by scale and present without meaningful quantization.
3. "The INT4 group-up repair is a compensation for activation shrinkage." Extension D: ρ = −0.17, p = 0.80.
4. "GPTQ is better because of error compensation." Calibration and compensation are confounded.
5. "The generic-sensitivity predictor measures quantization damage." It predicts the INT16 error as well (23–28 %).
6. "Language features matter for comprehension." Belebele is null.
7. Anything about deployed integer kernels or activation quantization. Nothing was tested.

---

## 6. Honesty audit

### 6.1 Statements that over-state the evidence

| # | document and statement | why it over-states | severity |
|---|---|---|---|
| O1 | `functional_validation/gemma9b/verdict.md`: "The Gemma-2-9B results replicate the Gemma-2-2B findings" and "consistent with the 2B observation that quantization damage concentrates on language-selective circuits" | Not independently reproduced. P3a is literally reversed in the freeze. P3b was redefined. The second clause contradicts the 2B rescue result. A consolidation note was added, but the text stands | high (if quoted) |
| O2 | `functional_validation/RESULTS_SUMMARY.md` F3 row and claim 4: "the language groups account for at most about 3 % of the quantization damage … at most 30 to 40 % of it sits in the layer-12 residual" | Repair shares are not upper bounds or locations (corrected in `PROJECT_RESULTS.md` §4.5 and `gptq_extension/RESULTS.md` §4.4 per D-110). The old wording remains in RESULTS_SUMMARY with only routing banners on top | medium |
| O3 | `PROJECT_RESULTS.md` §1 ("Two sets of tests were confirmatory: the three Phase 1 tests…"), §4.2 ("fixed before the run"), `phase1_results.md` ("pre-registered confirmatory tests"), `INDEX.md` ("pre-registered selective-versus-control tests") | V-10 / Q2: the OSF registration's freeze status is **unverified**, and "no pre-registration claim is established" (D-029). The tests come from the proposal; the estimands were the analyst's provisional choices. The course report says "apart from one family, all analyses are exploratory", which counts only Belebele. The documents disagree on whether Phase 1 is confirmatory | medium |
| O4 | `PROJECT_RESULTS.md` §4.10: "C bigger in 19 of 27 languages" (presented as support for context dominance) | The same 19 / 27 occurs at INT16 (VERIFIED). D-094's update says only the interval of the median difference is evidence | low–medium |
| O5 | `PROJECT_RESULTS.md` §1: "the part of it that looked real is explained by how strongly features fire" and "That gap is real, but it shrinks to nothing once you account for how strongly features fire" | True for the thresholded survival rate (A-039). Stage 1 shows a residual tier slope in the threshold-free λ after the same covariate, present at INT16 | low–medium |
| O6 | `PROJECT_RESULTS.md` §1 Belebele row: "Removing a *random* set of 1,024 to 2,860 features lowers accuracy by 1.2 to 6.0 points" | The −6.0 is the *control* sets (other languages' selective features). The full random non-selective pool (2,860) gives −2.2. The pre-written P1 (≥ 5 points) was not met and P4 was reversed (D-069). The positive control is weaker than the sentence suggests | low |
| O7 | `PROJECT_RESULTS.md` §4.2: "With strength matched the gap reverses: +3 to +5 points at INT6" | Holds at S > 5 with 5–10 activation bins. With 3 bins it is +0.80 (n.s.). At the pre-set cutoff it is +1.35 to +3.12 (A-035) | low |
| O8 | `PROJECT_RESULTS.md` §4.10 / `RESULTS_SUMMARY.md` claim 18: "Can we predict the error for features we held out? … reduces the held-out error by 21.5 %" | Held out by feature only. About 88 % of held-out documents are in calibration. On 136 items from 65 unseen documents the INT4 gain is 6.7 % (no interval) (`novelty_assessment_2026-10-26.md`). This caveat is absent from `PROJECT_RESULTS.md` | medium |
| O9 | `PROJECT_RESULTS.md` §5: "The group matters for reading comprehension — not found (and at most ~0.5 point could hide)" | No equivalence analysis was run for accuracy. The S0 interval is in nats per item. The auditor's approximate language-bootstrap bound is [−0.43, +0.57] pp for K64 but [−0.95, +0.77] pp for K = all. Reasonable but informal | low |

### 6.2 Statements that under-state the evidence (or omit relevant facts)

| # | item | comment |
|---|---|---|
| U1 | The F5 / Stage 3 / survival-effect convergence is spread across five documents and never stated as the project's strongest result | `PROJECT_RESULTS.md` leads with "why the findings are not conclusive". The measurement-validity finding is consistent across six analyses, two models (for one analysis) and three quantizers. It deserves more weight |
| U2 | F4 "no systematic difference" | Too strong in the other direction: the INT4 mean without Lao is +0.0032 [+0.0026, +0.0038] (formally non-zero, tiny, and partly present at FP) |
| U3 | Survival-effect noisy stratum | Two of six cells meet the 10 % rule (15.6 %, 15.0 %; n = 75). Not in `PROJECT_RESULTS.md`. Secondary by the freeze, but relevant for "where survival might matter" |
| U4 | Belebele INT5 / INT4 macro E intervals exclude 0 (−0.0066 and +0.0042) | Not mentioned in the plain-language page (probably noise or artefact; exploratory) |
| U5 | 9B median INT4 ratio 1.586 | The verdict reports only the outlier-driven mean (2.479). The median is the fairer comparison with 2B (1.54). It makes 9B look *more* similar to 2B than the verdict's number does |

### 6.3 Known corrections recorded by the project (all add-only, history preserved)

| correction | entry |
|---|---|
| Matched-RTN INT4 survival 18.7 % → 18.6 % (mean 18.649); GPTQ INT4 Belebele 0.319 → 0.318 (mean 0.31847) | Update to D-104 / D-107 (2026-10-29 01:13) |
| Deck line "only about 5 % of items clear the numerical floor" was wrong: 2,203 of 6,464 (34.1 %) | Update to D-094 (21:47) |
| "Preserved at INT6" → "changes little" (cosine 0.984, 1.9 floors); context dominance only for the *fixed* edit | Update to D-094 |
| D-054 (4) wrong: INT16 / INT12 are not identical to FP; the F5 floor must include INT16 | A-042, D-055 (before any F5 number) |
| `issues.md` B1 said TransformerLens computes RMSNorm in bf16; wrong (fp32) | `audit_our_vs_duan.md` line 45 |
| First S6 launch computed all-token statistics, against the proposal's equal budgets; cancelled before any output | A-026 → D-015 |
| F3 job submitted with an unanalysable preset; cancelled after 4 min | A-041 |
| TOST crashed with an empty bootstrap; fixed with no change to results | A-033 |
| Stage 3 selectivity comparison group "contaminated" in the first coding; fixed before the run | D-123 (3) |
| Repair-share wording ("damage sits in…") corrected to "repair achieved by this patch" | D-110; `PROJECT_RESULTS.md` §4.5; GPTQ RESULTS §4.4 |
| Causal-claims wording of `PROJECT_RESULTS.md` revised; old text archived | `archive/causal_wording_2026-10-27/` |
| Hardware mismatch (RTX 4070 vs 2080 Ti) in the extension-C INT4 job; rerun | D-079 |
| Study G extension D design time "17:20" is nominal; the exact commit (17:12:42) precedes module and run | D-084 time note |

### 6.4 Places where documents disagree or are stale

| # | documents | disagreement |
|---|---|---|
| S1 | `gptq_extension/RESULTS.md` line 3 vs its §4.6–4.8 and §7 | The header says "Study G, F5 and the external panel are still running"; the body reports all three as done |
| S2 | `PROJECT_RESULTS.md` §1, §4.9, §4.12, §5 vs its own top banner and `functional_validation/gemma9b/` | "no Gemma-2-9B results here" vs imported 9B results (superseded by the banner, text unchanged) |
| S3 | `functional_validation/RESULTS_SUMMARY.md` "Cannot be claimed" and §4 vs claim 17 | "Llama … planned … has not been run" vs the finished Llama check |
| S4 | `survival_effect_rescue/stage4_gemma_report_2026-10-28.md` §6 vs `survival_effect_rescue/README.md` | "The Llama-3.2-1B check … has not been run" vs D-114 done (a generated report frozen in time) |
| S5 | `PROJECT_RESULTS.md` §1 ("two confirmatory sets") vs `paper/course-report/main.tex` abstract ("apart from one family, all analyses are exploratory") vs `RESULTS_SUMMARY.md` ("Belebele, the only confirmatory family") | Phase 1's status is inconsistent (see O3) |
| S6 | `PROJECT_RESULTS.md` §4.10 / external README (|C| 0.133, |W| 0.073; INT6 "reversed") vs appendix A16 of the same file (median of the 6: C 0.333, W 0.096) | Lower median (analysis code) vs conventional median (table generator) for n = 6; at INT6 the two conventions give opposite orders of the medians |
| S7 | `PROJECT_RESULTS.md` appendix A11 note ("not the pooled figure of the text (INT4 12.7 / 18.6 / 30.5 %)") vs `gptq_extension/RESULTS.md` §4.1 ("Averages over the 27 languages") | 12.7 / 18.6 / 30.5 are language means, not pooled. Three INT6 survival aggregates coexist in the project: 94.9 % (pooled features), 90.4 % (median language), 85.0 % (mean language). All are correct, but they are easy to confuse |
| S8 | `PROJECT_RESULTS.md` §4.6 (attributes AUROC 0.506 to "F4") vs `results/f5/report.md` | 0.506 is F5's descriptive statistic S2c computed on the F4 pairs |
| S9 | 9B freeze P3a vs verdict / report | Inequality direction reversed (see 4.15) |
| S10 | Llama plan P3b (INT4 / FP amplification at ×0.088) vs 9B freeze P3b ("median selectivity ratio") | Same label, different quantity |
| S11 | `CURRENT_STATUS.md` header ("Updated 2026-10-27 14:34") vs body updates to 2026-10-29 | Cosmetic. `steering_feasibility_review_2026-10-28.md` also notes CURRENT_STATUS was partly stale |

### 6.5 Verification of headline numbers

"Recomputed" means the value was computed here from a committed CSV or JSON. "Re-read" means it was checked against a generated report file but not recomputed (e.g. permutation p-values).

| # | claim (document) | claimed | found | file | status |
|---|---|---|---|---|---|
| 1 | Pooled survival INT6 / INT4 (PR §4.2) | 94.9 / 13.75 % | 94.937 / 13.747 % | `phase1/mask/eq/summary.csv` | VERIFIED (recomputed) |
| 2 | Perplexity change INT6 / INT4 (PR §4.2) | +2.67 / +104 % | +2.674 / +103.88 % | same | VERIFIED |
| 3 | T1 INT7 (PR §4.2) | −0.28 pp [−0.63, +0.13], Holm 0.514 | same | `phase1/mask/eq/analysis.md` | VERIFIED (re-read) |
| 4 | T2 INT6 | −1.44 pp [−2.65, −0.13], Holm 0.128 | same | same | VERIFIED (re-read) |
| 5 | T3 tier slope | −0.0048, Holm 0.990; TOST ±0.02 | same | same | VERIFIED (re-read) |
| 6 | Median per-language INT6 survival (PR A2) | 90.4 % | 90.43 % | `phase1/mask/eq/per_language.csv` | VERIFIED (recomputed) |
| 7 | Tier means INT6 (PR §1) | 69.4 / 90.6 / 95.1 % | 69.4 / 90.6 / 95.1 | same + `g0/phase1_languages.json` | VERIFIED |
| 8 | Association retention INT8 … INT4, INT16 (PR §4.3) | 98.1 / 97.5 / 94.7 / 89.1 / 70.8; 99.2; lost 124 / 168 / 356 / 726 / 1,945 of 6,663 | identical from counts | `association_retention/results/eq/language_summary.csv` | VERIFIED (recomputed) |
| 9 | F1 macro and median (PR §4.4) | 0.704 [0.684, 0.723]; 0.262; 27 / 27 | 0.7037; 0.2624; 27 with lower bound > 0 | `functional_validation/results/eval/effects_by_language.csv` | VERIFIED |
| 10 | F1 median control harm (RS) | 0.024 | 0.0235 | same | VERIFIED |
| 11 | F1 on the 18 Belebele languages (RS claim 1) | 0.713 | 0.7127 | same + `mc_eval/accuracy.csv` | VERIFIED |
| 12 | F2 retention INT4 mean / median; INT5 mean (PR §4.4) | 2.14 / 1.54; 1.13 | 2.136 / 1.538; 1.129 | `results/eval/retention.csv` | VERIFIED |
| 13 | F3 INT4 rescue: group / all / full (RS) | 2.6 / 17.4 / 39.3 % | 2.62 / 17.36 / 39.28 % | `results/rescue/rescue.csv` | VERIFIED |
| 14 | Belebele accuracy FP none / zero / control; INT5; INT4 (PR §1) | 0.426 / 0.426 / 0.425; 0.387; 0.236 | 0.4257 / 0.4255 / 0.4251; 0.3874; 0.2358 | `results/mc_eval/accuracy.csv` | VERIFIED |
| 15 | Belebele K = all (PR §1) | 0.4257 → 0.4209, controls 0.4226 | same | `results/mc_ksens/accuracy.csv` | VERIFIED |
| 16 | Belebele S0 (PR §1) | −0.0001 [−0.0035, +0.0031] | same | `results/mc_eval/statements.md` | VERIFIED (re-read) |
| 17 | Belebele positive control (PR §1) | 1.2–6.0 points | −1.20 (1,024 random), −2.19 (all random), −6.01 (controls at all) | `results/mc_posctrl/summary.md` | VERIFIED numbers; wording loose (O6) |
| 18 | F5 median W1 INT8 / 6 / 5 / 4; floor (PR §4.6) | 0.997 / 0.988 / 0.961 / 0.839; 0.949 | same | `results/f5/report.md` | VERIFIED (re-read) |
| 19 | Study G zero-ablation median by state (PR §4.8) | 0.262 / 0.268 / 0.266 / 0.271 / 0.281 / 0.432 | 0.2624 / 0.2681 / 0.2664 / 0.2709 / 0.2808 / 0.4316 | `gain_study/results/curves.csv` | VERIFIED (recomputed) |
| 20 | Study G FP down medians (PR §4.8) | 0.022 / 0.090 / 0.169 / 0.262 | 0.0223 / 0.0902 / 0.1687 / 0.2624 | same | VERIFIED |
| 21 | Study G ×11.3 FP / INT4 (PR §4.8) | 1.21 / 0.81 | 1.213 / 0.810 | same | VERIFIED |
| 22 | INT4 ×1.1 net harm (PR §4.8) | about −0.001 | −0.0010 (21 / 27 negative); group harm −0.0014 (RS uses group harm) | same | VERIFIED |
| 23 | Extension C ×1.1 / ×1.2 (PR §4.8, RS 13) | −0.00135 [−0.00197, −0.00032], 20 / 27; −0.0017, 18 / 27 | −0.00135, 20 / 27; −0.00169, 18 / 27 | `gain_study/extension_c/cells.csv` | VERIFIED (median recomputed; interval re-read) |
| 24 | Extension D (PR §4.8) | ρ −0.17, p 0.80, reliability 0.92 | −0.170, 0.804, 0.917 | `gain_study/extension_d/analysis.json` | VERIFIED |
| 25 | α* > 1 at INT4: legacy / matched / GPTQ (GPTQ RESULTS §4.7) | 21 / 13 / 7 of 27 | 21 / 13 / 7 | `gain_study/results/alpha_star.csv`, `gptq_extension/results/gain/*/alpha_star.csv` | VERIFIED |
| 26 | Llama P1, P2, P2b median, P3a (PR §1) | 0.048 [0.018, 0.173], 21 / 27; 1.010 / 1.024; 0.86 [0.74, 1.03]; 20 & 15 / 27 | same | `functional_validation/llama/analysis.json`, `llama_robust/robust.json` | VERIFIED |
| 27 | Transfer INT4 group: \|C\| \|W\| \|I\|, C > W, fixed, native, FP (PR §4.10) | 0.310 / 0.104 / 0.071; 19 / 27; 0.848; 0.433; 0.263 | 0.3100 / 0.1035 / 0.0712; 19; 0.848; 0.433; 0.263 | `intervention_transfer/results/main/group_per_language.csv` | VERIFIED |
| 28 | Transfer INT6 \|C\| / \|W\| (PR §4.10) | 0.0091 / 0.0092 | 0.0091 / 0.0092 | same | VERIFIED |
| 29 | C > W at INT16 (D-094 update) | 19 / 27 | 19 / 27 | same | VERIFIED (supports O4) |
| 30 | Feature level INT6 / INT4 cosine, normalized error, norm ratio, n (PR §4.10) | 0.984 / 0.768; 17 % / 66 %; 1.09; 2,203 items of 235 features | 0.9840 / 0.7683; 0.165 / 0.664; 1.0945; n 2,203, 235 features | `main/analysis_summary.json` | VERIFIED |
| 31 | Predictive validation INT4 / INT6 / INT16 (PR §4.10) | 21.5 [12.7, 29.4] / 10.9 [−1.2, 21.4] / 23.4 % | same | `main/predictive_validation.csv` | VERIFIED |
| 32 | R1 locked cross-corpus (PR §4.11) | INT4 19.9 [7.8, 31.3]; INT6 18.6 → 15.8 [−2.7, 28.7]; INT16 28.2 | same | `recovery/cross_corpus_locked.csv` | VERIFIED |
| 33 | R2 gradient vs NLL (PR §4.11) | 0.63 [0.46, 0.80] | 0.632 [0.457, 0.796] | `recovery/gradient_vs_nll.csv` | VERIFIED |
| 34 | R3 support-matched ratio (PR §4.11) | ×1.12 / ×1.14, 144 pairs | log2 0.165 / 0.183 (×1.12 / ×1.14), n 144 | `recovery/support_contrast.csv` | VERIFIED |
| 35 | R4 native vs fixed (PR §4.11) | 27 / 27; −0.351 [−0.358, −0.293]; residual +0.014, range −0.78 to +0.23 | same | `recovery/native_fixed_*.csv` | VERIFIED |
| 36 | External INT4 \|C\| / \|W\| (PR §4.10, external README) | 0.133 / 0.073 | conventional median of the 6: 0.333 / 0.096; lower median: 0.133 / 0.073 | `intervention_transfer/results/external/group_per_language.csv`; code `phase1_transfer_analysis.py` `weighted_median` | **MISMATCH (definition)**: both values appear in `PROJECT_RESULTS.md` (§4.10 vs A16) |
| 37 | GPTQ survival INT6 / INT4 and loss (GPTQ RESULTS §4.1) | 85.0 / 89.7 / 97.5; 12.7 / 18.6 / 30.5; 0.590 / 0.385 / 0.130 | 85.0 / 89.7 / 97.5; 12.68 / 18.65 / 30.51; 0.5904 / 0.3848 / 0.1296 | `gptq_extension/results/stage1/stage1_paired_per_language.csv` | VERIFIED |
| 38 | GPTQ tier gains at INT6 (low / high) and at INT4 (GPTQ RESULTS §4.1) | +25 / +4; +22 / +19 / +13 (high / mid / low) | +25.2 / +4.3; +21.7 / +18.9 / +12.9 | same | VERIFIED |
| 39 | Lowest language at INT6, legacy / GPTQ | 53 % / 92.7 % | 53.00 / 92.66 % | same | VERIFIED |
| 40 | Matched-RTN Phase 1 Holm p (GPTQ RESULTS §4.2) | 0.042 / 0.038 | 0.0422 / 0.0384 | `stage1/analysis_rtnm/analysis.md` | VERIFIED (re-read) |
| 41 | GPTQ transfer normalized error and floors (GPTQ RESULTS §4.3) | 0.664 / 0.572 / 0.476; 0.165 / 0.153 / 0.132; 1.88 / 1.75 / 1.52 | same | `stage2/transfer_feature_level.csv` | VERIFIED |
| 42 | GPTQ − legacy INT4 total harm (GPTQ RESULTS §4.3) | −0.46 [0.31, 0.62], 26 / 27 | −0.461 [−0.624, −0.309], 26 negative | `stage2/transfer_group_summary.csv` | VERIFIED |
| 43 | Ablation median ratio and mean E at INT4 per arm (GPTQ RESULTS §4.4) | 1.54 / 1.36 / 1.22; 0.935 / 0.900 / 0.779 | 1.538 / 1.355 / 1.223; 0.935 / 0.900 / 0.779 | `stage3/analysis_*_eval/` | VERIFIED |
| 44 | Rescue per arm (GPTQ RESULTS §4.4) | group 2.6 / 2.3 / 2.0; all 17 / 14 / 11; full 39 / 35 / 41 | 2.62 / 2.3 / 2.0; 17.4 / 13.7 / 10.8; 39.3 / 35.2 / 41.4 | `stage3/analysis_*_rescue/rescue.csv` | VERIFIED |
| 45 | Belebele INT4 no edit per arm (GPTQ RESULTS §4.5) | 0.236 / 0.291 / 0.318 | 0.2358 / 0.2912 / 0.3185 | `stage3/analysis_*_mc/accuracy.csv` | VERIFIED |
| 46 | Study G ×0.088 at INT4 per arm; GPTQ ×11.3 (GPTQ RESULTS §4.7) | 0.323 / 0.281 / 0.187; 1.15 | 0.323 / 0.281 / 0.187; 1.146 | `gptq_extension/results/gain/*/curves.csv` | VERIFIED |
| 47 | GPTQ F5 INT4 median W1 and share at floor (GPTQ RESULTS §4.6) | 0.839 / 0.875 / 0.923; 0.3 / 0.7 / 11.7 % | same | `gptq_extension/results/f5/*/report.md` | VERIFIED (re-read) |
| 48 | External panel INT4 cosine and floors (GPTQ RESULTS §4.8) | 0.754 / 0.808 / 0.876; 7.1 / 6.0 / 4.9 | 0.7541 / 0.8083 / 0.8758; 7.07 / 5.99 / 4.91 | `external/transfer_feature_level.csv` | VERIFIED |
| 49 | Survival-effect B over A range (survival_effect README) | 3.1–4.4 %, intervals above 0 | 3.07–4.44 % | `survival_effect_rescue/results/survival_effect_results.json` | VERIFIED |
| 50 | Stage 3 median gain at INT4 per arm (PR §4.13) | 0.594 / 0.672 / 0.823 | same | `survival_measurement/stage3/stage3_results.json` | VERIFIED |
| 51 | Stage 3 median ν by tier at INT16 | 0.0192 / 0.0179 / 0.0170 | 0.01920 / 0.01794 / 0.01701 | same | VERIFIED |
| 52 | 9B macro retention and P1 / P3a (9B verdict) | 1.003 / 1.006 / 2.479; 27 / 27; 27 / 27 | 1.003 / 1.006 / 2.479; 27 / 27; 27 / 27 (harm order) | `gemma9b/results/analysis/curves.csv` (derived) | VERIFIED against imported table; raw NOT CHECKABLE |
| 53 | Duan reproduction (PR §4.1) | 81.19 vs 81.26; 51.15 vs 51.30 | same | `duan_codebase_audit.md` | VERIFIED (re-read; raw outside scope) |
| 54 | Clean-room recomputation 89 / 89 (PR §4.2) | 89 / 89 | stated in `phase1_reproduction.md` | — | NOT CHECKABLE (needs cached raw runs) |
| 55 | Stage 3 reproduction to 3e-8; identity to 1e-14 | — | stated in the generated results | `stage3_results.md` | NOT CHECKABLE here (needs raw sums from the private dataset) |
| 56 | Corrected slip: matched-RTN INT4 survival | 18.6 (was 18.7) | 18.649 | `stage1_paired_per_language.csv` | VERIFIED |

**Tally.** 56 items listed. 53 VERIFIED: most were recomputed from CSV/JSON, and about a dozen (marked "re-read") were checked against generated report files only. 1 MISMATCH (#36: a median-convention discrepancy, with both values in the same document). 2 NOT CHECKABLE. No numeric error was found in any recomputed headline number. The documentation's problems are in wording, labels and staleness (sections 6.1–6.4), not in arithmetic.

---

## 7. What was tried that did not work or gave nothing conclusive, and why

| attempt | outcome | why (diagnosis) |
|---|---|---|
| Our own first survival pipeline (Phase 0) | Disagreed with Duan; retired | Regime difference (BOS) and q/k/v scale sharing (A-008), not a bug (D-009) |
| No-BOS regime on FLORES | Instrument broken (perplexity 48,864; noise-floor gate fails; sign flips) | Short sentences without an attention sink (hypothesis, A-029 / A-030) |
| Phase 1 confirmatory tests | Null | Real effect is small; INT7 at the ceiling (99.35 vs 99.82 %); broad "selective" class (74 %); magnitude confound |
| Behavioural link (selective damage → loss) | Rank association only | Collinear with overall damage (VIF 9.5); n = 27 languages |
| JumpReLU margin as a better magnitude variable | Added nothing | Thresholds nearly constant (3.00–3.19) |
| Belebele as the downstream task | Null | Blunt instrument (43 % accuracy, 385 items), 9 languages unusable; loss-task effects do not carry over (unexplained) |
| Belebele positive control prediction P1 (≥ 5 points) | Not met; P4 reversed | Random non-selective features are less consequential than expected; other languages' selective features (incl. English prompt-scaffold features) are more consequential |
| F5 rule S2 (status AUROC at INT4) | Undefined (legacy), not met (GPTQ, 0.49) | Floor effect: almost no feature stays within noise at INT4 under RTN |
| Study G repair hypotheses H-G5a / b and α* vs 1/g | Not met | Effect about 0.001 nats, under-powered; disappears under GPTQ |
| Study G H-G4 (precision ordering) | Narrowly not met | Threshold artefact: the INT8 median dips by 1 SE |
| Extension D (shrinkage mechanism) | Not supported | α* reliable (0.92) but unrelated to shrinkage |
| Extension B (state-defined groups more consequential) | Not met (opposite sign) | The features that matter do not change with precision as much as the selection does |
| Llama consistency | Not consistent | Too many confounds (model, SAE type, recipe, depth, tokenizer); smaller effects; the frozen mean estimator is fragile |
| Transfer UTILITY (main) | Not met at INT6 | INT6 error near the floor (1.9 floors); little to predict |
| Generic-sensitivity predictor as a quantization audit | Not quantization-specific; not cheap | Predicts INT16 equally; needs 4 control interventions per context; document leakage |
| Gradient attribution / survival as predictors of transfer | Add nothing at INT4 | Vector target vs scalar gradient (R2 shows scalars do track loss) |
| Survival R1 as a predictor of effect similarity | 3–4 % (target 10 %) | W1 hard to predict at all (baselines 0.2–3.2 % over null); R1 and W1 share noise |
| Gain and ν (new survival metrics) | Add nothing beyond R1 | Same as above; ν still carries peak size |
| H4 tier specificity under GPTQ | Inconclusive | Depends on standardisation; about 23 effective languages |
| Precision audit (R5) | Never run | User HOLD, then deadline (D-122) |
| GPTQ repeat of association retention / F4 | Not run | Disk (2.76 GB per arm vs 2.4 GB free) |
| 9B three-arm confirmation | Only legacy RTN exists, imported | Run by another team; GPTQ arms unfinished |

---

## 8. Limitations (consolidated)

1. **One primary model, one layer, one SAE.** Gemma-2-2B, layer 12, Gemma Scope 16k. The only second-model evidence is Llama (confounded, not consistent) and 9B (same family, unverified, RTN only).
2. **Simulated weight-only quantization.** bf16 / float execution with rounded weights. No integer kernels, no activation or KV quantization. INT16 is a real small perturbation, not an arithmetic-noise oracle. The paired precision audit was not run.
3. **Quantizer dependence.** Most magnitudes, and several verdict words (Phase 1, H-G4, UTILITY external, F2 class), change with the quantizer. Legacy RTN had a non-standard head-shared scale.
4. **27 languages, three tiers, about 23 effective clusters** for tier slopes (single-language script families). Tier is observational and entangled with script, tokenization, corpus size and model competence.
5. **FLORES reuse.** Selection on dev and evaluation on devtest, which had been examined in earlier studies. Hence almost everything after Phase 1 is exploratory. Fresh text exists only for extension C (reserve devtest), the 6-language external panel and R1.
6. **Pre-registration status.** Phase 1's registration is unverified. Belebele is the only family whose freeze-before-scoring is documented and labelled confirmatory by all documents.
7. **Multiplicity.** No correction across the many exploratory families or the three quantizer arms.
8. **Held-out validation.** Transfer prediction is feature-held-out with about 88 % document overlap. Survival-effect is language-held-out, not document-held-out. No external-panel F5 exists.
9. **Control quality.** Activity-matched controls are imperfectly norm-matched. Controls in the transfer study are not eligibility-filtered. The F5 swap control uses one control per feature, without intervals for the differences.
10. **Ceiling and floor effects.** INT7 survival at the ceiling. INT4 Belebele at chance under legacy RTN. F5 floor share capped at about 0.95. INT6 transfer error near the floor.
11. **Downstream scope.** One reading task, zero-shot, letter format, 18 languages, 385 items each.
12. **Hardware sensitivity** at the 1e-3 nat level (D-079). Comparable to the INT4 fine-gain effects.
13. **Sample design.** F5 is stratified, not random (1,487 of 4,382). Groups are fixed at K = 64.

---

## 9. Open questions, future work, and what is publishable

### 9.1 What is publishable now (candid view)

- **A measurement / negative-results paper** (workshop or findings-level). Thesis: *SAE feature "survival" under weight quantization measures feature size and a quantizer-wide shift, not functional continuity.* Evidence:
  - λ decomposition and ν (Stage 1 / 3);
  - survival-effect (3–4 %, two models);
  - F5 swap controls;
  - tag AUROC about 0.5;
  - three quantizers;
  - Belebele null with a positive control.

  This is coherent, well controlled and honest. Its weaknesses are one model, the FLORES reuse and simulated quantization. The novelty assessment (`novelty_assessment_2026-10-26.md`) warns that ingredients overlap with Duan v2, Borobia, FEGA, Chang et al. and Wu et al. So it needs careful positioning as a *controlled evaluation*, not a discovery.
- **Not publishable as is:**
  - the original selective-destruction hypothesis (null);
  - the INT4 "repair" (tiny, legacy-specific);
  - the context-vs-weights decomposition as a mechanism (estimand-dependent);
  - the generic-sensitivity predictor as an audit (not quantization-specific, document leakage);
  - any 9B claim before reproduction.

### 9.2 What would be needed for a stronger, confirmatory result

| need | concrete step | rough size |
|---|---|---|
| Confirmatory replication on fresh data | Freeze H3 / H5-type hypotheses (λ ~ size; R1 adds < X % beyond baselines; equivalence margin stated) and test on text never examined (e.g. a new Wikipedia or NTREX / FLORES+ sample), document-held-out | 27+ languages, ≥ 300 new documents per language |
| Second model with the same SAE family | Finish 9B (layer 20) with all three arms; reproduce the imported RTN results from raw files first (`gemma9b/REVIEW_STATUS.md` task 2) | 3 arms × INT6 / INT4 |
| Second SAE on the same model / layers | Gemma Scope widths 65k or other L0, and layers 6 / 18, to separate SAE and layer from model | 2–3 SAEs × 2–3 layers |
| Real kernels | GPTQ / AWQ INT4 with real integer kernels (e.g. Marlin / ExLlama) vs fake-quant on the same weights; run the precision audit R5 | small (paired) |
| Activation quantization | W8A8 / W4A8 to see whether feature survival behaves differently when activations are rounded | one extra arm |
| Tier question | More languages (Belebele and FLORES cover 100+), preregistered tier-vs-scale mediation design with an explicit causal question (D-032) | 60–100 languages for adequate power on a tier slope |
| Belebele equivalence | State an equivalence margin (e.g. ±1 point) and test it; add more languages and tasks (e.g. XNLI, generation with chrF) | ≥ 4× items to halve the about ±0.5 pp interval |
| GPTQ mechanism | GPTQ with off-language, small, or random calibration sets to separate compensation from calibration | 3–4 calibration variants |
| Multiplicity | Pre-specify the arm family and correct across arms | — |

### 9.3 Open questions the project itself lists

Q29 (run the precision audit), Q33 (answered), Q21 / Q23 / Q24 (archives, survival rule, full F5 sample), Q15 (why low-resource languages have lower activation scale; never designed). See `DECISIONS_FOR_REVIEW.md`.

---

## Appendix A. Glossary (simple words)

| term | meaning |
|---|---|
| SAE (sparse autoencoder) | A "dictionary" of 16,384 directions that describes what the model's middle layer is representing at each word. Each direction is a "feature" |
| feature fires / active | The feature's value is above its threshold (JumpReLU) on a token |
| survival | A feature "survives" quantization if its activations over many tokens still correlate at r ≥ 0.9 with full precision |
| R1 | That correlation itself, as a continuous number |
| λ (lambda) | Log noise-to-signal ratio of a feature after quantization. It splits the correlation into gain and noise relative to the feature's spread |
| gain g | How much of the feature's full-precision activation size is kept (1 = all) |
| ν (nu) | How far quantization pushes the feature's pre-activation, measured in units of its own threshold |
| flips | Tokens where the feature switches on or off because of quantization |
| selectivity / language-selective | A feature that fires much more on one language than on others (S(f) above a shuffled-label cutoff) |
| association retention / tag | Whether a feature still has its original language as its clear peak after quantization |
| margin | How far above the runner-up language a feature's peak language was in full precision (borderline vs clear) |
| matched controls | Features chosen to fire about as often (and as strongly) as the language group but not language-specific; the fair comparison |
| zero-ablation | Set the features to zero and measure the change in loss |
| net harm / effect E | Loss increase from removing the group minus that from removing matched controls (nats per token) |
| retention (functional) | Net harm in the quantized model divided by net harm in full precision (1 = unchanged) |
| rescue / patching | Put back full-precision activations of chosen parts into the quantized model and measure how much of the extra loss disappears |
| W1 / effect fingerprint | The change in output probabilities when one feature is switched off; W1 is the cosine between its full-precision and quantized versions |
| swapped-direction control | Ablate a *different* feature's direction at the same places and strengths, to see whether the change is feature-specific |
| numerical floor | How much a measure moves from tiny rounding alone (INT16); effects near it cannot be trusted |
| gain study / up-down scaling (Study G) | Multiply a language's group by a factor below 1 (turn down) or above 1 (turn up) and measure the loss |
| α* | The best scaling factor for a language (lowest loss) |
| intervention transfer | Does a fixed edit made in the full-precision model have the same effect in the quantized model? |
| context (C) / weights (W) / interaction (I) | Parts of the change in an edit's effect due to quantizing the layers before it, the layers after it, and both together |
| fixed vs native edit | Remove the feature by its full-precision amount vs by the amount it actually has in the quantized model |
| RTN (legacy) | Round each weight to the nearest allowed value; the project's original version shared one scale across attention heads |
| matched RTN | Standard round-to-nearest with one scale per output channel and GPTQ's grid |
| GPTQ | A quantizer that rounds weights one by one and adjusts the remaining weights to cancel the error, using calibration text |
| INT8 … INT4 | Number of bits per weight (fewer = smaller and rougher); INT16 / INT12 = almost no change; INT0 = restore original (identity check) |
| BOS / mask regime | The start-of-text token; "mask" = it is added but excluded from all statistics |
| Holm | Correction of p-values for running several tests together |
| TOST / equivalence | A test showing an effect is smaller than a stated margin, not just "not found" |
| AUROC | How well a score separates two groups (0.5 = not at all, 1 = perfectly) |
| cluster bootstrap | Resampling whole languages (or sentences) to get honest intervals when items within them are related |
| freeze | A committed file fixing the analysis rules before the data are scored |
| confirmatory / exploratory | Fixed in advance and reported whatever the outcome / everything else |
| mutation check | Deliberately breaking the analysis code to confirm the tests catch it |

## Appendix B. Study → decision entries → key files

| study | D / A entries | key result files | plan / spec |
|---|---|---|---|
| Duan reproduction, regime | D-001, D-002, D-009, D-027, D-028; A-001–A-024, A-029, A-030 | `audit_our_vs_duan.md`, `duan_codebase_audit.md`, `g0/` | — |
| Phase 1 survival | D-003, D-010–D-026, D-032–D-034; A-026–A-040; V-10 | `phase1/mask/eq/{summary.csv, per_language.csv, analysis.md}`, `phase1_results.md`, `phase1_summary.md`, `phase1_reproduction.md` | `anlp_revised.txt` |
| Association retention + A1–A3 | D-035–D-038, D-045, D-066, D-067 | `association_retention/results/{eq, within-language, margin-function}/` | `docs/superpowers/plans/2026-10-21-language-association-retention.md` |
| F1 / F2 loss task | D-039–D-043, D-068 | `functional_validation/results/eval/` | `docs/superpowers/specs/2026-10-21-functional-validation-design.md` |
| F3 rescue | D-058; A-041 | `functional_validation/results/rescue/` | same |
| F4 | D-046, D-048, D-049, D-051 | `functional_validation/results/f4/` | same |
| F5 | D-053–D-057, D-059–D-064; A-042 | `functional_validation/results/f5/`, `f5_sensitivity/` | `docs/superpowers/specs/2026-10-23-feature-function-equivalence-design.md` |
| Belebele | D-044, D-047, D-050, D-062, D-069, D-072, D-086 | `functional_validation/results/{mc_eval, mc_ksens, mc_posctrl}/`, `promptcheck/`, `evaluation_freeze_mc.json` | functional-validation plan |
| Study G + B / C / D | D-070–D-085 | `functional_validation/gain_study/{results, extension_c, extension_d, state_defined}/` | `docs/superpowers/specs/2026-10-24-graded-gain-study-design.md` |
| Llama-3.2-1B | D-087, D-089, D-090 | `functional_validation/llama/`, `llama_robust/` | `docs/superpowers/plans/2026-10-26-llama-consistency-check.md` |
| Intervention transfer + R1–R4 (+ R5 unrun) | D-088, D-094–D-100 | `intervention_transfer/results/{main, external, recovery}/` | `docs/superpowers/specs/2026-10-26-intervention-transfer-research-design.md` |
| GPTQ / matched-RTN extension | D-101–D-111, D-115, D-116 | `gptq_extension/results/{stage1, stage2, stage3, f5, gain, external}/`, `gptq_extension/RESULTS.md` | `docs/superpowers/plans/2026-10-27-gptq-matched-comparison.md` |
| Survival vs effect | D-110, D-112–D-114 | `survival_effect_rescue/` | `docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md` |
| Survival measurement validity | D-117–D-124 | `survival_measurement/{stage1, stage3}/`, `analysis_freeze.json` | `docs/superpowers/plans/2026-10-29-survival-measurement-validity-study.md` |
| Gemma-2-9B (imported) | D-089, D-092, D-125 | `functional_validation/gemma9b/` | `docs/superpowers/plans/2026-10-26-gemma9b-confirmation.md` |
| Novelty / publication assessments | D-065, D-088 | `novelty_assessment_2026-10-26.md`, `publication_reassessment_2026-10-26/`, `steering_feasibility_review_2026-10-28.md` | `docs/superpowers/plans/2026-10-26-publication-recovery-after-results.md` |

*End of audit. Scratch scripts used for the checks: `/tmp/opus_analysis/verify.py` and inline commands; no repository file other than this one was written.*

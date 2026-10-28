# Plan: a measurement-validity study of SAE feature "survival" under weight quantization

Date: 29 October 2026, 04:09 IST. Written by Claude at the user's request ("lets add testing metric 1 and 2 to the plan ... check if [Codex's direction] is worthwhile and add to plan. lets work towards this").
Status: **PLAN ONLY. Stage 1 (CPU, existing files) waits for the user's go-ahead on this plan; Stage 2 (one GPU pass) needs a separate go-ahead (Q33).** Nothing has been computed for it yet.
Scope: Gemma-2-2B, Gemma Scope layer-12 16k SAE, the three weight quantizers already run (legacy RTN, matched RTN, GPTQ), BOS-masked Phase 1 text. No new model, layer, SAE or quantizer.

## 1. The direction, and whether it is worth it

Codex proposed (29 October): "The strongest supported paper direction is now a measurement-validity study: quantization method strongly changes reported survival rates, while preserved activation patterns give only limited information about preserved feature effects. The analyses do not support the stronger story that damaged language features explain the model's language-performance losses or provide a successful repair."

**Checked against the committed results; every part holds.**

| part of the claim | evidence | file |
|---|---|---|
| quantizer strongly changes reported survival | INT6 85.0 / 89.7 / 97.5 %, INT4 12.7 / 18.6 / 30.5 % (legacy / matched RTN / GPTQ); the 6-bit low-resource-tier gap mostly belongs to the legacy quantizer | D-104, `reports/gptq_extension/RESULTS.md` 4.1 |
| activation agreement gives limited information about preserved effects | R1 adds 3 to 4 % held-out error reduction for the effect similarity W1 (target 10 %) in all six cells on Gemma, and on Llama-3.2-1B; firing-set overlap and scale ratio add nothing beyond R1 (model C, -0.5 to +0.4 %); the retained/lost label scores AUROC 0.49 | D-113, D-114, D-108; `reports/survival_effect_rescue/stage4_gemma_report_2026-10-28.md` sections 3-4 |
| damaged language features do not explain the language losses | the FP-donor group patch recovers 2.6 / 2.3 / 2.0 % of the INT4 loss gap; Belebele null in every arm, also at an above-chance GPTQ INT4 model | D-058, D-107 |
| no successful repair | Study G's INT4 boost "repair" falls from 21 to 7 of 27 languages under GPTQ (= full precision); pre-set repair hypotheses not met in every arm | D-109 |

**Why it is worth it.** It is the strongest claim the complete evidence supports, it uses the nulls as results rather than hiding them, and it answers the future work Duan v2 names (section 7: causal validation of feature changes). **Why it is not enough alone.** "Survival does not establish function" is already published for pruning and for SAE feature recovery (Borobia, Neurocomputing 2026; arXiv 2607.12166), and "the steering direction survives quantization" for CAA vectors (arXiv 2609.06473). A measurement-validity paper needs a positive methodological piece: **why** the survival rate misleads and **what** to report instead. Metrics 1 and 2 below are that piece. Search record: section 8.

## 2. The two metrics

Notation, per (language, feature j, quantized state): a = full-precision activation over tokens, b = quantized activation on the same tokens, θ_j = the JumpReLU threshold (`reports/phase1/sae_thresholds.csv`, N11), pre_j = the encoder pre-activation.

**Metric 1: decomposition of the survival correlation.** Write b = g·a + c + e with g = Cov(a,b)/Var(a) (gain), c the offset and e the remainder. Then the survival correlation is exactly

  r = g·σ_a / sqrt(g²·σ_a² + σ_e²),  so  λ := log(1/r² − 1) = log(σ_e² / (g²·σ_a²)).

λ is the log noise-to-signal ratio of the feature. Three consequences follow from algebra, not from data: (i) r is blind to g (a tenfold shrink leaves r = 1; this is already noted in the 26 October addendum); (ii) r > 0.9 means σ_e < 0.484·g·σ_a, so for the same perturbation a feature with a larger spread passes more easily; (iii) a fixed cut-off turns a common change in the noise level into a large and nonlinear change of the share above 0.9. Reported alongside λ: g (all tokens and co-active tokens), the firing-set flip rates (on→off, off→on) and c.

**Metric 2: perturbation in threshold units.** ν_j = RMS over FP-active tokens of (pre_j,Q − pre_j,FP) / θ_j. It measures how hard the quantizer pushed the encoder direction of feature j, in the units in which JumpReLU decides on/off, without dividing by the feature's own activation spread. Paired with it: the margin m_t = (pre_j,FP,t − θ_j)/θ_j and the share of active tokens with |Δpre| > margin (predicted flips).

**What is already known about these, so it is not re-claimed.** Firing-set overlap and the co-active scale ratio (parts of metric 1) exist for the 1,487 F5 features and add nothing to predicting W1 (model C, D-113). So the plan does **not** claim that metrics 1 or 2 predict preserved effects better; that is tested only as a secondary check (H5). The claims under test are about what the survival rate itself measures.

## 3. Hypotheses (written before any number of this plan exists)

- **H1 (one sensitivity per feature).** If quantizers differ mainly in their overall noise level, λ of the same (language, feature) differs between arms and bit widths by a near-constant shift: high rank agreement and a slope near 1.
- **H2 (the survival rate is a threshold on a shifted distribution).** Each arm's per-language survival rate is reproduced from the legacy-RTN λ distribution plus one shift per (arm, bits).
- **H3 (scale artifact).** λ depends strongly on the feature's full-precision activation scale; ν (metric 2) does not, or much less.
- **H4 (language and tier).** After metric 2 removes feature scale, does the perturbation still differ by resource tier or by language selectivity? Either answer is a result: "no" means the tier and selectivity patterns in survival are a measurement property of the statistic (consistent with N6, N9); "yes" means a genuine differential perturbation that survival rates had mixed with scale.
- **H5 (secondary, expected null).** g and ν added to model B do not reach the 10 % target for W1. Reported to bound the claim, not to rescue it.

## 4. Stages

### Stage 1: CPU only, existing files (needs only the go-ahead on this plan)

Inputs: `survival.pt` (eq and all) for legacy (`~/.cache/anlp-always-late/phase1_runs/p1-mask/`), matched RTN (`transfer_runs/rtnm-p1-mask/`) and GPTQ (`transfer_runs/gptq-p1-mask/`); each holds `corr` (27 × 16,384) for bits 8, 7, 6, 5, 4, 16, 12, 0 and FP `firing_rate`, `mean_activation`, `max_activation`. The variance of a, g and σ_e are **not** saved, so Stage 1 can test H1, H2 and the scale part of H3 using λ only, with max and mean activation as scale proxies.

1. Reproduction control first: the committed per-language survival rates (A2, D-104 tables) recomputed from these files must equal the committed tables.
2. Compute λ per (arm, bits, language, feature) on active features (FP firing rate > 0.001, Duan's rule). r ≥ 0.9999 and r ≤ 0 are reported as censored counts, not dropped silently; the analysis uses ranks where censoring matters.
3. H1: Spearman of λ between arms at INT6 and INT4, per language, and the Deming slope of λ_GPTQ on λ_legacy. H2: predicted versus observed survival rate per language from a single shift per (arm, bits). H3 (proxy): share of λ variance explained by log max activation and firing rate per arm.
4. Re-run the N9 tier regression with λ instead of the thresholded survival rate, per arm.
5. Labelled **post hoc and exploratory**: λ is a transform of survival values that earlier analyses (N6, N9, D-104) already used.

Script `scripts/g0/survival_lambda.py`, test first and mutation-checked as the project requires; results in `reports/survival_measurement/stage1/`.

### Stage 2: one GPU pass for the sufficient statistics (needs Q33)

A **separate sidecar** run through `scripts/g0/gptq_ext/run_with_weights.py`; the hash-pinned entry points and frozen code are not edited. Same Phase 1 text (BOS masked, equal budget), same tokens, all 27 languages, all 16,384 features. States: FP, INT16 (floor), INT6, INT4 in the three arms (INT16 and FP are shared). Streamed per (language, feature, state), float64:
- n, Σa, Σb, Σa², Σb², Σab over all tokens and over co-active tokens;
- counts FP-on/Q-off and FP-off/Q-on;
- Σ Δpre and Σ Δpre² over FP-active tokens, and the count with |Δpre| > distance to θ_j.

Gates, in order: INT0 gives identical statistics to FP; the correlation recomputed from the sums equals the committed `survival.pt` corr (to float tolerance) in every arm; smoke test, then a timing and memory test before the full job; check disk on Ada before running (no deletion). Raw outputs stay outside git; small tables are committed; the private HF dataset gets the rest.

### Stage 3: frozen analysis of Stage 2 (freeze file written and committed before the Stage 2 outputs are read)

`reports/survival_measurement/analysis_freeze.json` fixes populations, transforms, clustering (language as cluster, 2,000 bootstrap replicates) and the decision rules. **Proposed thresholds (to be confirmed or changed by the user or Codex before the freeze; Q33):**
- H1 metric-1 identity: r recomputed from g, σ_a, σ_e equals the measured r within 1e-4 for all features (a correctness gate, not a finding).
- H3 met if, within each (arm, bits), |Spearman(ν, log FP scale)| < 0.2 while |Spearman(λ, log FP scale)| > 0.5.
- H4: language-level regression of mean log ν on tier with script as covariate (as in N8/N9); "no tier effect" only by an equivalence bound fixed in the freeze, never from a non-significant p-value.
- H5: the D-113 rule unchanged (10 % relative MAE reduction over B, cluster interval above zero).

### Stage 4: write-up

One plain-language report generated from the result files (generator plus test, as for `PROJECT_RESULTS.md`), a decision entry and a handoff. The paper's frame: (1) reported survival rates depend strongly on the quantizer and on feature scale, and a fixed cut-off mixes both; (2) metric 1 shows why, metric 2 removes scale; (3) activation agreement in any of these forms carries limited information about preserved effects; (4) the language-damage and repair stories are not supported. Keep the course report (due 31 October) as it is unless the user asks for a paragraph from Stage 1.

## 5. What this plan cannot show

One model, one layer, one SAE, simulated weight-only quantizers. λ is exact algebra but its interpretation as "noise" assumes the remainder e is unrelated to a; JumpReLU gating breaks that near the threshold, which is why Stage 2 measures flips and ν directly. Stage 1 reuses seen data. No deployment, kernel or activation-quantization claim.

## 6. Order and dependencies

Stage 1 → (Q33) Stage 2 gates → full Stage 2 → freeze → Stage 3 → Stage 4. If Stage 1 shows H1 and H2 hold almost perfectly, Stage 2 still runs (it is the only way to separate g, σ_e, flips and ν); if Stage 1 cannot reproduce the committed tables, stop and report.

## 7. Files

- This plan; `reports/survival_measurement/` (stage results, freeze, report); `scripts/g0/survival_lambda.py` and the Stage 2 sidecar with tests in `scripts/g0/tests*/`.
- Decision D-117 (this plan), question Q33 (GPU go-ahead and thresholds).

## 8. Prior-art check (29 October 2026, targeted, not exhaustive)

| work | what it does | overlap with this plan |
|---|---|---|
| Duan v2, [arXiv 2606.03002](https://arxiv.org/html/2606.03002v2) | Pearson survival, bands 0.9 / 0.5, damage score 1 − r, peak activation predicts survival (AUC 0.92-0.97) | parent; **no** decomposition, no attenuation analysis, no feature matching, no causal metric (checked in the v2 text) |
| Borobia, [Neurocomputing 2026](https://www.sciencedirect.com/science/article/pii/S0925231226022241), [arXiv 2603.25325](https://arxiv.org/html/2603.25325v1) | pruning; decoder-cosine matching, firing-rate quintiles, single-feature KL ablation | survival ≠ local function (taken); no decomposition, no metric-artifact analysis, no quantization experiments in the arXiv version |
| [arXiv 2607.12166](https://arxiv.org/pdf/2607.12166) | correlational recovery overstates causal validity | general form of claim (3) is taken |
| [Gupte et al. 2025](https://arxiv.org/abs/2507.15977) | full-precision SAEs transfer to pruned models | transfer level only, no per-feature metric |
| [Bhandari and Wade, arXiv 2609.06473](https://arxiv.org/abs/2609.06473) | steering under INT8/NF4; vectors stay collinear | steering, not SAE features; no per-feature activation metric |
| [arXiv 2606.12138](https://arxiv.org/html/2606.12138), [Paulo and Belrose](https://arxiv.org/pdf/2501.16615) | feature instability across seeds; subspace-level reproducibility | alternative matching metrics, not perturbation metrics |
| [JumpReLU, arXiv 2407.14435](https://arxiv.org/pdf/2407.14435) | threshold as the on/off scale | basis of metric 2; no flip-rate or margin robustness study found |

Not found in the sources checked: a decomposition of SAE survival correlation into gain, noise and flips; an attenuation (noise-to-signal) account of why peak activation predicts survival; a threshold-unit perturbation measure. "Not found" is not "first"; repeat the search before submission.

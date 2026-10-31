# Always Late: Complete Findings

**Does weight quantization damage the language-specific features a sparse autoencoder finds in a language model?**

Updated 31 October 2026 after the Gemma-2-9B three-arm confirmation.
Every number below comes from committed analysis files; cross-references are in parentheses.

---

## 1. What We Did

We took a language model (Gemma-2-2B), looked inside it with a sparse autoencoder (SAE) that splits the hidden state into 16,384 interpretable features, and asked: when you compress the model's weights to fewer bits, what happens to the features that are specific to individual languages?

We tested three things that "happens to" could mean:
- **Activation survival:** does the feature still fire the same way on the same tokens? (the existing measure from earlier work)
- **Functional effect:** does removing the feature still harm the language's text prediction by the same amount? (our contribution)
- **Identity:** can you still tell which feature it is from what it does to the output? (our contribution)

We ran these tests on 27 languages at bit widths from 8 down to 4, with three different quantizers, on two Gemma model sizes, and compared with a Llama model.

### Models and SAEs

| model | SAE | layer | features | hook | role |
|---|---|---|---|---|---|
| Gemma-2-2B | Gemma Scope JumpReLU 16k | 12 of 26 (46%) | 16,384 | `blocks.12.hook_resid_post` | main study |
| Gemma-2-9B | Gemma Scope JumpReLU 16k | 20 of 42 (48%) | 16,384 | `blocks.20.hook_resid_post` | confirmation |
| Llama-3.2-1B | TopK SAE (k=48) | 12 of 16 (75%) | 14,336 | `blocks.12.hook_resid_pre` | consistency check |

### Quantizers

| method | description |
|---|---|
| Legacy RTN | round-to-nearest with shared attention scales across heads (from the paper we replicated) |
| Matched RTN | standard RTN: one scale per output channel, same grid as GPTQ |
| GPTQ | calibrated rounding using 8,192 Wikipedia tokens per language (27 languages, 221k total) |

All quantization is simulated (weights rounded but stored as floats; no integer kernels, no activation quantization).

### Text and languages

27 FLORES-200 languages across three resource tiers (9 high, 9 mid, 9 low). Feature selection on dev split; evaluation on devtest (300 sentences per language). Comprehension test: Belebele (385 items per language, 18 languages that passed calibration). Second corpus: fresh Wikipedia articles in 6 languages.

---

## 2. The BOS Correction

**We found and corrected a methodological problem in the earlier Gemma evaluation (Duan 2026).**

We reproduced Duan's reported survival rates with their released code (INT7: 81.19% vs 81.26%, INT6: 51.15% vs 51.30%). The disagreement with our pipeline traced to a missing beginning-of-sequence (BOS) token in their Gemma input protocol.

| metric | without BOS | with BOS |
|---|---|---|
| FP perplexity (WikiText) | 458.4 | 16.1 |
| INT6 survival | 51.15% | 97.52% |
| INT7 survival | 81.19% | 99.97% |

(`reports/audit_our_vs_duan.md`, `reports/duan_codebase_audit.md`, D-009)

This is a large input-protocol effect. It does not establish that every conclusion in Duan's paper is wrong (the feature populations differ between settings), but it means their reported survival numbers cannot be taken at face value for Gemma.

We also retired our own first pipeline after finding separate flaws (counting BOS in feature statistics, changing tokens during decode/re-encode). Both corrections are documented in the decision log.

---

## 3. Main Findings on Gemma-2-2B

### 3.1 The language features matter for text prediction

Removing each language's 64 most selective features (zero-ablation) harms that language's next-token loss far more than removing activity-matched control features, in all 27 languages.

- Macro net harm: **0.704 nats/token** [0.684, 0.723]
- Median per-language net harm: 0.262 nats/token
- Positive in **27 of 27** languages

(`results/eval/report.md`, D-041)

**Status:** exploratory (evaluated on previously examined FLORES devtest).

### 3.2 The dependence survives quantization — and grows at INT4

| state | macro retention (E_state / E_FP) |
|---|---|
| INT8 | 1.01 |
| INT7 | 1.03 |
| INT6 | 1.04 |
| INT5 | 1.13 |
| **INT4** | **2.14** |
| INT16 | 1.00 |
| INT12 | 1.00 |
| INT0 | 1.00 (exactly) |

The effect is flat through INT5 and roughly doubles at INT4. The INT4 amplification happens because the quantized model is *already damaged* — the denominator (FP effect) stays the same while the numerator (INT4 effect) grows because quantization itself harms the model.

**Critical caveat:** The 2.14 retention does NOT mean the features carry the quantization damage. It means the features' *relative* importance grows when the model is degraded. See section 3.4.

(`results/eval/report.md`, D-041, D-050)

### 3.3 Graded dose-response (Study G)

Scaling a language's features up or down produces a graded, specific and consistent response across all 27 languages.

| hypothesis | result |
|---|---|
| H-G1: dose-ordered harm (down) at FP | **met** (27/27 strictly ordered) |
| H-G2: specificity (FP) | **met** (26/27 lower bound > 0) |
| H-G3: dose-ordered harm (up) at FP | **met** (27 increasing, 24 net positive) |
| H-G4: same ordering across precisions | **not met** (median clause failed) |
| H-G5a: generic INT4 repair | **not met** |
| H-G5b: group-specific INT4 repair | **not met** (12/27, threshold 14) |
| H-G6: consistency across states | **met** (27/27 every state, rank corr 0.936) |

(`gain_study/results/report.md`, D-078)

The threshold at INT4 is real: harm is flat from FP through INT5, then jumps (median zero-ablation net harm: 0.26–0.28 at FP–INT5, then 0.43 at INT4).

**Extension C** (562 fresh sentences, 5 control draws): at INT4, turning the group up by 1.1 lowers loss by a median 0.00135 nats/token (20/27 languages). This is about 0.05% of the loss — a measurable but tiny effect, not a practical repair method. (`gain_study/extension_c/report.md`, D-082)

**Extension D:** the idea that the repair compensates for the group's activation shrinkage is **not supported** at the language level (rank correlation with 1/g_ratio: −0.17). (`gain_study/extension_d/report.md`, D-084)

### 3.4 The damage does NOT sit in the language features

Patching full-precision activations back into the language groups recovers very little of the quantization loss:

| what is restored | share of INT4 loss gap recovered |
|---|---|
| Language group (K=64) | **2.6%** |
| Matched controls | 1.2% |
| All SAE features | 17.4% |
| Whole layer-12 residual | 39.3% |

At INT5 and INT6, the language group's share is ~0%. The features account for at most 3% of the damage. Most quantization damage is distributed across the model, not concentrated in language-specific features.

(`results/rescue/report.md`, D-058)

### 3.5 A feature's effect stays recognizable even when its activation doesn't

At INT4 under legacy RTN, the activation-based survival criterion (r > 0.9) says only 13.2% of features "survive." But by output effect:

| state | median cosine (effect fingerprint) | share within noise | identity retrieval |
|---|---|---|---|
| INT8 | 0.997 | 94% | 0.997–1.000 |
| INT6 | 0.988 | 90% | |
| INT5 | 0.961 | 66.5% | |
| INT4 | 0.839 | **0.3%** | **0.997** |

Even at INT4, 99.7% of features are still individually identifiable from their effects, even though their activations have changed.

**But the INT4 similarity loss is generic.** A swapped-direction control (ablating a *different* feature's direction at the same positions and amplitudes) gives nearly the same cosine at INT4: 0.828 vs 0.842. The gap between the real feature and the control is only 0.003 to 0.014 at every precision. This means the loss of similarity reflects how the downstream computation changes under quantization for *any* perturbation, not anything specific to the feature.

(`results/f5/report.md`, D-059, D-061, D-063, D-064)

**This is one of the most important findings in the project:** activation survival and functional effect preservation are measuring different things. The activation correlation cutoff (r > 0.9) misclassifies about 31% of functionally intact features at INT5.

### 3.6 Association retention is mostly a margin effect

Whether a feature keeps its preferred language after quantization is predicted by how "borderline" it was in full precision:

| what predicts loss? | AUROC |
|---|---|
| Margin → association loss at INT6 | **0.90** |
| Margin → association loss at INT5 | 0.81 |
| Margin → association loss at INT4 | 0.75 |
| Margin → **function** change at INT6 | 0.58 |
| Margin → function change at INT5 | 0.53 |

The margin barely predicts functional change (Spearman with W1 ≈ +0.10). Association status and activation survival are not reliable proxies for functional continuity.

(`association_retention/results/within-language/report.md`, `margin-function/report.md`, D-066, D-067)

### 3.7 Belebele: the features do not matter for reading comprehension

**This is the only confirmatory (pre-registered) family in the project.**

Zeroing a language's group does not change Belebele accuracy beyond matched controls:

- Macro effect at FP: −0.0001 nats/item [−0.0035, +0.0031]
- Accuracy: 0.426 without edit, 0.426 group zeroed, 0.425 controls zeroed
- With the whole group zeroed (up to 1,719 features): same null

The selected groups are too small to produce a measurable task-level effect on this test. The edit path itself is sensitive (random sets of 1,024–2,860 features do lower accuracy), so this is not a methodological failure — the groups just don't matter enough for Belebele.

(`results/mc_eval/statements.md`, `results/mc_ksens/report.md`, D-047, D-050, D-062)

### 3.8 State-defined groups (Extension B)

The 64 most selective features change with precision (Jaccard with FP selection: 0.86 at INT6, 0.49 at INT4). But groups selected in the quantized model are NOT more consequential in that model than the FP selection. The dose-response curves coincide. The INT4 repair appears with both selections.

(`gain_study/state_defined/results/report.md`, D-085)

---

## 4. Three Quantizers Compared (GPTQ Extension)

On Gemma-2-2B, comparing legacy RTN, matched RTN and GPTQ:

### 4.1 Survival

| state | legacy RTN | matched RTN | GPTQ |
|---|---|---|---|
| INT6 | 85.0% | 89.7% | **97.5%** |
| INT4 | 12.7% | 18.6% | **30.5%** |

GPTQ improves INT4 survival by 17.8 pp over legacy RTN. Matched RTN alone (better implementation, same method) accounts for 5.9 pp.

### 4.2 The original confirmatory tests (T1–T3)

The original confirmatory tests asked whether language-selective features survive worse than non-selective ones. Under the original RTN, the answer was null (T2 at INT6: p = 0.128, Holm-corrected). Under GPTQ, the answer is also null. The selective-fragility question does not have a positive answer under any quantizer.

(`gptq_extension/RESULTS.md`, D-101 to D-110)

---

## 5. Gemma-2-9B Confirmation (Three Arms)

Run on 31 October 2026 on an RTX PRO 6000 (Blackwell), all seven units completed with zero failures.

### 5.1 Tier 1 (Study G): **CONSISTENT**

| prediction | rule | 9B result | met? |
|---|---|---|---|
| Q0: existence and grading in matched RTN and GPTQ | P1 at FP + H-G1 at INT4 | both arms met | **yes** |
| Q1: INT4 amplification smaller with GPTQ | r(GPTQ) ≤ r(matched), r(GPTQ) ≤ r(legacy)−0.3 | legacy 1.689, matched 1.490, GPTQ **1.165**; GPTQ below legacy in 26/27 languages | **yes** |
| Q2: 6 bits the same everywhere | retention in [0.85, 1.15] all arms | legacy 1.006, matched 1.040, GPTQ 1.015 | **yes** |
| Q3: repair hint | H-G5b per arm | not met in any arm (8, 3, 2 of 27) | reported |

(`out/analysis/g9-compare/report.md`, `amendment_A1_three_arm.json`)

### 5.2 Tier 2 (Phase 1 survival and loss): **CONSISTENT**

| prediction | rule | 9B result (legacy / matched / GPTQ) | met? |
|---|---|---|---|
| S1: survival ordering | GPTQ > matched > legacy, GPTQ above legacy in ≥24 languages | INT4: 22.3% / 32.9% / **46.6%**; 27/27 languages | **yes** |
| S2: loss ordering | GPTQ < matched < legacy, GPTQ below legacy in ≥24 languages | INT4: 0.410 / 0.287 / **0.088** nats/token; 27/27 | **yes** |
| S3: tier gap at INT6 | gap smaller under GPTQ | 9.1 / 6.3 / **1.7** points | **yes** |
| S4: original tests | Holm-corrected p | T2 at INT6 below 0.05 in legacy and matched; null in GPTQ | reported |

### 5.3 Legacy arm predictions (from the original 2026-10-26 freeze)

All five P1–P3b predictions met. Macro retention at INT4: 2.479 (2B: 2.14). Dose-response ordering: 27/27 at FP.

### 5.4 Limits of the 9B confirmation

- Scale replication within one architecture (Gemma 2) and one SAE family (Gemma Scope). One model, one layer, one dictionary.
- The 2B GPTQ reference values used in the comparison are exploratory (D-101 to D-110), not independently verified.
- The legacy RTN results were pre-seeded from HF (adapter hash `8cc54217`); K64 groups identical, `all` family Rs differ.
- No hardware bridge: 9B ran on Blackwell (RTX PRO 6000) vs 2B on Ada (A100). Bit-identical FP reference does not exist for 9B.
- The 9B verdict.md's interpretation that "damage concentrates on language-selective circuits" is an overstatement of what amplification alone shows (see section 3.4).

---

## 6. Llama-3.2-1B Consistency Check: NOT CONSISTENT

The Llama check used a different model family, SAE architecture (TopK vs JumpReLU), SAE recipe, training data, relative depth (75% vs 46%), and tokenizer.

| prediction | result |
|---|---|
| P1: existence at FP (≥24/27) | **not met** (21/27, median E = 0.048 vs Gemma 0.262) |
| P2a: flat through INT6 | met (INT8 1.010, INT6 1.024) |
| P2b: INT4 amplification | **not met** (median ratio 0.86 vs Gemma 1.54) |
| P3a: graded dose-response | **not met** (20/27 at FP, 15/27 at INT4) |

**Verdict: not consistent.**

**What this means and does not mean.** The group effect exists in most Llama languages (21/27) but is small. The INT4 amplification is absent. But model family, SAE architecture, recipe, training data and tokenizer all change together — this is not a controlled comparison. A negative verdict here does not show that the Gemma findings fail on other models. The confirmation that separates model from SAE architecture is the Gemma-2-9B check (which is consistent).

(`llama/report.md`, D-087, D-090)

---

## 7. Intervention-Transfer Study (D-088)

This study asks: when a feature's effect changes under quantization, what explains the change? It decomposes the transfer error into context effects (C), weight effects (W), and their interaction (I) using a crossed prefix/suffix design.

**Status:** core design frozen, 404 features with FLORES and 6-language Wikipedia panel, main runs and external runs completed on Ada, CPU recovery analysis done. Gemma-2-2B only (per amendment A1). Results exist in `reports/intervention_transfer/results/`.

This study was not part of the original proposal scope and is complete but was run on the compute cluster; its results are committed and referenced from `CURRENT_STATUS.md`. It is distinct from the functional validation work and contributes the decomposition and diagnostic pieces.

---

## 8. What Can Be Claimed — and What Cannot

### Can be claimed (with stated limits)

1. **Language-selective SAE features matter for text prediction** across 27 languages in both Gemma sizes. The effect is robust: graded, specific to the language group, and consistent across precisions from FP through INT5.

2. **INT4 amplifies the ablation effect**, but this reflects the model's degradation, not damage concentrating in those features. The rescue study shows the features account for at most ~3% of the INT4 loss.

3. **GPTQ substantially improves feature survival and reduces changes in feature effects** compared to RTN, on both model sizes.

4. **Activation survival and functional preservation are different.** The standard r > 0.9 cutoff misclassifies about a third of functionally intact features at INT5. At INT4, almost no feature passes activation survival but 99.7% keep their identity.

5. **The INT4 loss of similarity is generic,** not feature-specific. A swapped-direction control changes by the same amount.

6. **Association retention is mostly a margin statistic** — how borderline a feature already was predicts its loss, but that margin barely predicts functional change.

### Cannot be claimed

- That the groups matter for downstream comprehension (the Belebele test is null).
- That the results generalize to other model families (the Llama check is "not consistent," though confounded).
- That the results hold for other layers, SAEs, or activation quantization (one layer, one SAE, weights-only throughout).
- That quantization damage is located in language features (the rescue study directly refutes this framing).
- That the INT4 similarity loss tells us something specific about individual features (the direction-swap control shows it doesn't).
- Anything confirmatory beyond the Belebele family (which is a null).

---

## 9. Unfinished and Stale Work

### Unfinished branches

| branch | status |
|---|---|
| `feature/extensions-on-main` | E1 (firing-fidelity) and E2 (overlap analysis) **code only, never run, no results.** 1 commit beyond main. |
| `feature/local-final-pipeline` | **Stale.** Branched 355 main-commits ago. Its RQ0 claim (quantization axis) is wrong (the cause was missing BOS, not the axis). Contains AI-generated paper text violating course rules. **Should not be merged.** |

### Known defects documented internally

- **9B verdict.md** overinterprets amplification as causal ("damage concentrates on language-selective circuits") — the consolidation note flags this.
- **H-G4** (precision ordering of down-ratio harm) narrowly failed; within two bootstrap errors it passes. Reported honestly as not met.
- **H-G5** (repair by turning features up at INT4) not met in any arm, any model. The repair effect is real but too small and inconsistent to qualify.
- **Llama perturbation norms** are mismatched at INT4 (group 0.12 vs controls 0.19), which complicates interpretation.

---

## 10. Key Numbers at a Glance

| measurement | Gemma-2-2B | Gemma-2-9B |
|---|---|---|
| Languages tested | 27 | 27 |
| Selective features (K=64) | 8,086 | 5,218 |
| FP net harm (median, nats/tok) | 0.262 | — |
| INT6 retention | 1.04 | 1.006 |
| INT4 retention (legacy) | 2.14 | 2.479 |
| INT4 survival (legacy / matched / GPTQ) | 12.7 / 18.6 / 30.5% | 22.3 / 32.9 / 46.6% |
| INT4 loss (legacy / matched / GPTQ, nats/tok) | — | 0.410 / 0.287 / 0.088 |
| Rescue: language group share of INT4 loss | 2.6% | — |
| Belebele: group effect at FP | −0.0001 nats/item | — |
| F5 effect identity at INT4 | 99.7% retrieval | — |
| Llama P1 existence | 21/27 (not met) | — |

---

## 11. Honest Summary

The project's strongest finding is methodological: activation survival, language association, and functional effect are three different things, and they come apart under quantization. The standard activation cutoff (r > 0.9) substantially overestimates functional damage, while the INT4 loss of output similarity reflects the model's general numerical drift rather than feature-specific damage.

The substantive finding — that language-selective SAE features carry a measurable, graded dependence for text prediction in 27 languages, robust across Gemma model sizes and stable through moderate quantization — is real but comes with important limits. The features account for very little of the actual quantization damage (~3%), they don't matter for reading comprehension, and the pattern doesn't clearly replicate on Llama (though that comparison is confounded by many simultaneous changes).

GPTQ's improvement over RTN is the most practically relevant finding: it roughly halves the INT4 amplification effect on 9B (ratio 1.165 vs 1.689) and more than doubles INT4 survival (46.6% vs 22.3%). This holds across both model sizes.

The project is internally consistent. Most analyses are exploratory. The one confirmatory family (Belebele) produced a null. Every failure and limitation is documented.

---

## Source Files

| what | where |
|---|---|
| 2B claim map with evidence links | `reports/functional_validation/RESULTS_SUMMARY.md` |
| 2B gain study | `reports/functional_validation/gain_study/results/report.md` |
| 9B three-arm comparison | `out/analysis/g9-compare/report.md` |
| 9B legacy verdict | `reports/functional_validation/gemma9b/verdict.md` |
| 9B artifact hashes | `reports/functional_validation/gemma9b/artifact_hashes.json` |
| Llama report | `reports/functional_validation/llama/report.md` |
| BOS correction | `reports/audit_our_vs_duan.md` |
| Association retention | `reports/association_retention/results/eq/association_retention.md` |
| Decision log | `reports/decisions_and_anomalies.md` |
| Intervention transfer | `reports/intervention_transfer/results/` |
| Course report (LaTeX) | `paper/course-report/main.tex` |
| wandb runs | entity `karma-skz-`, project `anlp-project` |

# Always Late: The Complete Picture

**Does compressing a language model damage its internal language-specific features?**

Written 2026-10-31. This document collects every finding of the project across all branches and studies into one place. It is a map for people who want the whole story; every number here comes from the files linked in the source column. One model family (Gemma 2), one SAE family (Gemma Scope), weights-only round-to-nearest quantization (and, later, GPTQ), 27 FLORES-200 languages, one layer per model size.

---

## 1. What the project asks

Large language models are compressed (quantized) for deployment. Sparse autoencoders (SAEs) decompose a model's internal representations into thousands of interpretable features, some of which are "language-selective" — they fire strongly and specifically for text in one language. The project asks:

1. Do those language-selective features still matter for their language after quantization?
2. Do low-resource languages lose more?
3. Does a downstream reading task (Belebele) care?
4. Does a single feature still do the same thing?
5. Can you repair the damage by turning the features back up?
6. Does any of this depend on which quantizer you use?
7. Does any of this hold on a larger model?

---

## 2. Setting

| dimension | value |
|---|---|
| Primary model | Gemma-2-2B, layer 12 of 26 |
| Confirmation model | Gemma-2-9B, layer 20 of 42 |
| Supporting model (negative) | Llama-3.2-1B, layer 12 of 16 |
| SAE | Gemma Scope 16k (JumpReLU); Llama uses a TopK SAE (width 2048, k=48) |
| Languages | 27 FLORES-200 languages across three resource tiers (9 low, 9 mid, 9 high) |
| Quantizers | Legacy RTN (original), matched RTN (standard), GPTQ (calibrated on Wikipedia in 27 languages) |
| Bit widths | FP, INT16, INT12, INT8, INT7, INT6, INT5, INT4 |
| Features per language | K=64 most selective, with activity-matched controls (3 draws) |
| Sentences | 300 held-out FLORES devtest per language (Study G also uses 562 reserve sentences) |
| Regime | BOS masked, equal token budgets (27,036 tokens per language) |

---

## 3. The findings

### 3.1 The features matter (F1, Study G)

**Zero-ablating a language's 64 most selective features harms that language's next-token loss far beyond matched controls, in all 27 languages.**

- Macro harm beyond controls: 0.704 nats/token [0.684, 0.723]; median 0.262
- Matched-control harm: median 0.024 — an order of magnitude smaller
- Positive in 27 of 27 languages, at every group size tested (K=16, 64, all)
- The group carries about 4.8% of its language's activation mass (range 2–23%)

Confirmed on Gemma-2-9B (27/27 languages, P1 met). On Llama-3.2-1B the effect exists but is weaker (median E = 0.048 vs 0.262; 21/27 positive, threshold of 24 not met).

### 3.2 The effect survives quantization — and grows at INT4 (F2)

**The language-group effect is preserved through INT5 and amplified at INT4.**

| precision | macro retention (effect / FP effect) | interpretation |
|---|---|---|
| INT16 | 1.00 | rounding noise floor |
| INT12 | 1.00 | rounding noise floor |
| INT8 | 1.01 | preserved |
| INT7 | 1.03 | preserved |
| INT6 | 1.04 | preserved |
| INT5 | 1.13 | preserved |
| INT4 | 2.14 | amplified |

The INT4 amplification means removing the features at 4 bits hurts twice as much as at full precision — because the quantized model is already damaged and removing the features compounds that damage.

Confirmed on Gemma-2-9B: retention ladder is 1.003 / 1.047 / 1.006 / 1.205 / 2.479 at INT8 through INT4. Same shape, same INT4 amplification.

On Llama-3.2-1B: P2a met (INT6 flat at 1.024) but P2b **not met** (INT4 retention is -3.57, dominated by one language with a tiny FP effect).

### 3.3 The quantization damage is NOT in the language features (F3)

**Patching full-precision activations back into the quantized model at layer 12 shows the language groups account for at most ~3% of the quantization-induced loss.**

| what is patched | share of INT4 loss gap recovered |
|---|---|
| Language group (64 features) | 2.6% (controls: 1.2%) |
| All SAE features | 17.4% |
| Whole layer-12 residual | 39.3% |

Most of the quantization damage arises in the layers above, not in the language-selective features themselves. This holds across all three quantizers (legacy 2.6%, matched 2.3%, GPTQ 2.0%).

### 3.4 Low-resource languages: a quantizer effect, not a feature effect

**Phase 1 confirmatory tests (T1–T3): none of the three pre-registered tests rejects its null after Holm correction under legacy RTN.**

- T1 (INT7): -0.28 pp, Holm p = 0.514
- T2 (INT6): -1.44 pp, Holm p = 0.128
- T3 (tier slope): equivalent to zero within ±0.02

Language-selective features do NOT survive worse than matched controls. The equivalence test supports "no tier effect larger than 0.02 in median r."

**However, the null is fragile.** Under matched RTN, T1 and T2 cross 0.05 (Holm 0.042 and 0.038). Under GPTQ, nothing is significant. The effect is at most 1.5 percentage points near a ceiling — tiny whatever the quantizer.

**The tier gap (low-resource vs high-resource survival) is mostly a quantizer artifact.** At INT6, the gap is 25.7 points under legacy RTN but only 4.8 points under GPTQ. Activation magnitude explains the language-level survival differences (Spearman correlation of activation scale with survival: +0.97).

### 3.5 Belebele reading task: no effect (confirmatory)

**The only confirmatory family: zeroing a language's feature group does NOT change Belebele accuracy beyond matched controls.**

- S0 not met: macro effect at FP is -0.0001 nats/item [-0.0035, +0.0031]
- S1–S5: undefined (no language shows a clearly positive FP effect)
- Accuracy: 0.426 without edit, 0.426 group zeroed, 0.425 controls zeroed
- At INT4 under legacy RTN the model is at chance (0.236); under GPTQ it scores 0.319 (above chance) and **still shows no group effect** — the null is not an artifact of a broken model
- The frozen prompt scores higher than the lm-evaluation-harness prompt in all 18 covered languages

### 3.6 Does a single feature still do the same thing? (F5)

**Through INT8, yes (within noise). At INT4, the effect changes — but each feature is still itself.**

| precision | median W1 (cosine of effect fingerprints) | share within noise floor | identity retrieval |
|---|---|---|---|
| INT16 | 0.997 | 95.0% | 1.000 |
| INT8 | 0.997 | 94.0% | 1.000 |
| INT6 | 0.988 | 89.7% | 1.000 |
| INT5 | 0.961 | 66.5% | 1.000 |
| INT4 | 0.839 | 0.3% | 0.997 |

**The INT4 change is generic, not feature-specific.** Ablating a *different* feature's direction at the same positions gives the same cosine drop (0.828 vs 0.842). The drift sits in the quantized downstream computation, not in the features themselves. Each feature remains individually identifiable (retrieval 0.997–1.000 at every precision).

**The activation-based survival metric is only partly informative about function:** at INT5, 82% of surviving features are functionally within noise vs 46% of those that do not survive, but 31% of functionally intact features are counted as "lost."

With GPTQ at INT4: 11.7% of features stay within noise (vs 0.3% under legacy RTN).

### 3.7 Study G: graded gain — scaling the features up and down

**The dependence on the language features is graded, specific, and consistent across 27 languages.**

| hypothesis | verdict |
|---|---|
| H-G1: dose ordering when turning down | **met** (27/27 languages at FP) |
| H-G2: turning down is specific to the language group | **met** (27/27 lower bound above 0) |
| H-G3: turning up also costs | **met** (27 increasing, 23 net positive) |
| H-G4: harm grows with fewer bits | narrowly **not met** |
| H-G5a/b: turning up repairs INT4 damage | **not met** |
| H-G6: consistency across languages | **met** (27/27, rank correlation >0.5) |

**The precision dependence is a threshold at INT4:** the dose-response is flat from FP through INT5, then the zero-ablation harm jumps from 0.28 to 0.43 nats/token at INT4 and the down-ratios become 1.9× larger.

**Extension C (562 fresh sentences, 5 control draws):** a small INT4 repair replicates — turning the group up by 1.1× lowers the loss by 0.00135 nats/token (20/27 languages) — but it is tiny (~0.05% of the loss) and precision-specific. The same turn-up raises the loss at FP and INT6.

**Extension D:** the per-language optimal gain at INT4 is reliable (split-half 0.92) but NOT ordered by how much the group's activation shrank (rank correlation -0.17). No compensation mechanism is supported.

**Extension B (state-defined groups):** groups selected in the quantized model are NOT more consequential in that model than the full-precision selection (H-Gb2 not met, opposite sign). The full-precision selection does not drive the INT4 results.

**Under GPTQ:** the INT4 amplification shrinks from 1.9× to 1.1× full precision. The repair hint disappears (best factor >1 in 7/27 languages, same as FP). The pre-set repair hypotheses stay "not met" in every arm.

### 3.8 The association margin predicts label loss, not function loss

- Within-language AUROC of the feature's full-precision margin for losing its association: 0.90 (INT6), 0.81 (INT5), 0.75 (INT4)
- AUROC of the same margin for the *function* event: 0.58 (INT6), 0.53 (INT5)
- Correlation of the margin with W1: about +0.10 (weak)

**The association label and the activation survival fraction are not reliable proxies for functional continuity.** At the feature level, functional continuity cannot be told apart from the model's response to any direction.

---

## 4. The GPTQ comparison (three-arm study)

Three quantizers, same model, same features, same tests. Written for readers who did not follow the technical details.

### 4.1 Feature survival and loss

| bits | survival: legacy / matched / GPTQ | loss: legacy / matched / GPTQ |
|---|---|---|
| INT6 | 85.0% / 89.7% / **97.5%** | 0.0254 / 0.0197 / **0.0058** |
| INT4 | 12.7% / 18.7% / **30.5%** | 0.590 / 0.385 / **0.130** |

GPTQ is better than legacy RTN in 27/27 languages at both bit widths. About a third of the gap between legacy RTN and GPTQ is closed by simply doing RTN the standard way.

### 4.2 What holds, what changes

| finding | under GPTQ |
|---|---|
| Removing the language group hurts that language | **holds** (27/27) |
| Harm amplified at INT4 | **holds, smaller** (1.22× vs 1.54×) |
| Patching features back repairs ~3% of the loss gap | **holds** (2.0%) |
| Belebele: no group effect | **holds**, now also at above-chance INT4 |
| F5: features stay identifiable | **holds**, 11.7% within noise at INT4 (vs 0.3%) |
| Study G: INT4 repair hint | **disappears** (7/27, same as FP) |
| Low-resource tier gap at INT6 | **nearly closes** (25.7 → 4.8 points) |

### 4.3 What was fragile

The Phase 1 null (selective vs control survival): verdict depends on the quantizer in a delicate way, effect tiny in all cases. Part of the legacy RTN's weakness was implementation-specific (shared scale across attention heads, slightly different rounding grid).

---

## 5. The Gemma-2-9B confirmation (D-089)

Predictions frozen before any 9B number existed. Same SAE family (Gemma Scope), layer 20 of 42, width 16k. Three arms: legacy RTN, matched RTN, GPTQ.

### 5.1 Tier 1 (Study G): **consistent**

| prediction | 9B result | met? |
|---|---|---|
| Q0: existence and grading under new quantizers | Both arms: P1 met, H-G1 met | **met** |
| Q1: INT4 amplification smaller with GPTQ | GPTQ 1.165 vs legacy 1.689; GPTQ below legacy in 26/27 | **met** |
| Q2: 6 bits are the same in every arm | Legacy 1.006, matched 1.040, GPTQ 1.015 | **met** |
| Q3: repair hint | Legacy 14/27, matched 8/27, GPTQ 4/27 above 1 | reported |

Legacy-freeze predictions (P1–P3b): all met. Label: **consistent**.

### 5.2 Tier 2 (Phase 1): **consistent**

| prediction | 9B result | met? |
|---|---|---|
| S1: survival ordering GPTQ > matched > legacy | INT4: 22.3% / 32.9% / 46.6%; GPTQ > legacy in 27/27 | **met** |
| S2: loss ordering GPTQ < matched < legacy | INT4: 0.410 / 0.287 / 0.088 nats/token | **met** |
| S3: tier gap smaller under GPTQ | Gap: 9.1 / 6.3 / 1.7 points | **met** |
| S4: T1–T3 per arm | Reported (legacy null, matched significant, GPTQ null) | reported |

### 5.3 Limits

One architecture (Gemma 2), one SAE family, one layer per model size, weights-only rounding. Different GPU (RTX PRO 6000 Blackwell) and torch build from the 2B runs. No 2B-to-9B bridge (bit-identical FP numbers).

---

## 6. The Llama-3.2-1B check: not consistent

Pre-registered verdict: **not consistent.** The group effect exists but is weaker, the INT4 amplification is not found, and the graded dose-response holds in only 20/27 languages at FP. This is confounded by model family, SAE architecture (TopK vs JumpReLU), SAE recipe, depth, and tokenizer. It is not a replication — it is a minimum consistency check that failed.

---

## 7. What can be claimed, and what cannot

### Can be claimed (with limits)

1. Language-selective SAE features are necessary for their language's next-token loss (27/27 languages, two model sizes)
2. That necessity survives quantization through INT5 and is amplified at INT4
3. The amplification and the resource-tier gap shrink dramatically with a better quantizer (GPTQ)
4. The quantization damage mostly does NOT sit in the language features (~3% of the loss gap)
5. Each feature stays individually identifiable at every precision (retrieval ≥0.997)
6. The change in a feature's effect is generic — not feature-specific — at every precision
7. Removing the language group does not change Belebele reading accuracy (confirmatory null)
8. The association-retention label does not predict functional continuity
9. There is a small, precision-specific repair at INT4 when turning the group up by 1.1× (~0.001 nats/token)

### Cannot be claimed

- That the groups matter for a downstream comprehension task (the Belebele null)
- That results hold for other models, layers, quantizers, or activation quantization
- That the Llama result supports or contradicts the Gemma findings (too many confounds)
- That the INT4 "repair" is practically useful (it is 0.05% of the loss)
- That the association or survival labels are reliable proxies for function
- That the Phase 1 selective-vs-control null is robust (it is fragile across quantizers)

---

## 8. The studies at a glance

| study | scope | status | key result |
|---|---|---|---|
| Phase 1 (survival and loss) | 2B, 27 languages, 7 bit widths | confirmatory + exploratory | null: selective features survive no worse; fragile across quantizers |
| F1 (necessity at FP) | 2B, 27 languages | exploratory, frozen setting | positive in 27/27 languages |
| F2 (across precisions) | 2B, 27 languages, 7 bit widths | exploratory, frozen setting | preserved through INT5, amplified at INT4 |
| F3 (rescue / where the damage sits) | 2B, 27 languages, INT4/5/6 | exploratory | ≤3% of the loss gap in the language features |
| F4 (lost vs retained association) | 2B, 26 languages | exploratory | no difference at equal activity |
| F5 (single-feature fingerprints) | 2B, 1487 features | exploratory | generic drift, identity preserved |
| Belebele (reading task) | 2B, 18 languages | **confirmatory** | null: no group effect |
| Study G (graded gain) | 2B, 27 languages, 6 states | exploratory | graded, specific, consistent; INT4 threshold; repair tiny |
| Study G ext. B (state-defined groups) | 2B, 27 languages | exploratory | FP selection does not drive INT4 results |
| Study G ext. C (repair with larger sample) | 2B, 27 languages, 562 sentences | exploratory | repair replicates at 1.1× only, size ~0.001 nats |
| Study G ext. D (optimal gain vs shrinkage) | 2B, 27 languages | exploratory, post hoc | not supported (ρ = -0.17) |
| GPTQ extension (three-arm) | 2B, 27 languages, 3 quantizers | exploratory | direction holds, magnitudes change; tier gap closes |
| Llama-3.2-1B check | 1B, 27 languages | exploratory | **not consistent** |
| Gemma-2-9B confirmation | 9B, 27 languages, 3 quantizers | exploratory, frozen predictions | **consistent** (tier 1 and tier 2) |
| Belebele prompt cross-check | 2B, 18 languages | exploratory | frozen prompt scores higher than harness prompt |
| Association margin (A2/A3) | 2B, 1487 features | exploratory | margin predicts label loss (AUROC 0.90), not function (0.58) |

---

## 9. Branch and repository map

| branch | what it contains |
|---|---|
| `main` | All 2B studies: functional validation (F1–F5), Phase 1, Belebele, Study G (main + extensions B/C/D), rescue, association retention, Llama-3.2-1B check, GPTQ three-arm extension, prompt cross-check, wandb v2 upload |
| `gemma9b-confirmation` | 9B three-arm confirmation: adapter, calibration, driver, freeze, amendment, results (tier 1 consistent, tier 2 consistent) |
| `feature/extensions-on-main` | Pilot results (FP/INT16/INT8/INT4 ladder), firing-fidelity sink (E1), non-survived overlap analysis (E2), freeze helper |
| `feature/local-final-pipeline` | Final submission: ACL paper draft, viva deck, matched survival, causal restoration, protocol reconciliation |

---

## 10. One paragraph for the paper

Language-selective SAE features in Gemma-2 produce a clear, graded, language-specific effect on next-token loss that survives weight quantization through INT5 and is amplified at INT4, across 27 languages and two model sizes (2B and 9B). The amplification and the low-resource disadvantage at moderate bit widths are largely artifacts of the legacy quantizer: under GPTQ both nearly disappear. The quantization damage itself sits mostly outside these features (≤3% of the loss gap at layer 12). Each feature remains individually identifiable at every precision, but its change in effect is generic — a swapped direction changes just as much — so functional continuity at the feature level cannot be separated from the model's overall numerical drift. The confirmatory downstream test (Belebele reading comprehension) finds no effect of removing the language group, including at 4-bit precision where the GPTQ model is above chance. A small repair (turning the features up by 1.1× at INT4) replicates on fresh sentences but is practically negligible (~0.001 nats/token) and disappears with a better quantizer.

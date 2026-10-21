# Prior-art check for the functional study (D-040), 2026-10-21

Question checked: has anyone studied (i) feature-level causal effects of language-selective SAE feature groups, (ii) compared across weight-quantization levels, (iii) with full-precision-donor patching to localise quantization damage by language, (iv) with multilingual task scores, and (v) whether association retention predicts functional retention?

Method and trust. One subagent searched and read primary pages (arXiv abstract, HTML or PDF; ACL Anthology; ICLR pages) on 2026-10-21 and returned a report. **I re-read two of its central claims myself** (the abstract page of 2603.25325 and the full text of 2609.23125, both confirmed below). Everything else is **agent-reported** and marked so in the register (P30 and later); venue statements taken from arXiv comments are not proceedings checks. Search coverage is limited and the field moves fast: **re-check before submission.**

## Verdict

No paper was found that combines all five elements. Several papers pre-empt parts of the method, and one is a scoop risk:

| work | what it covers | what it does not |
|---|---|---|
| **Duan, arXiv 2606.03002 (v2, 4 Jun 2026; "submitted to TMLR")** (P01) | Same model, SAE, layer and RTN ladder as ours; per-feature survival (Pearson) on WikiText-2; names causal validation as future work | no interventions, no languages. **Scoop risk:** the causal step on the same setup could appear any time; our differentiators are the language dimension, D-035 association retention and F3 to F4 |
| **Sathyanarayanan, arXiv 2609.23125 (19 Sep 2026, preprint)** (P30; re-read by me) | Activation quantisation of the residual stream at one block (not weights); the oracle of restoring chosen SAE features to full precision: restoring the 128 most damaged features gives perplexity 0.567 of the quantised model against 0.964 for random features (agent-reported numbers, confirmed in the full text by my re-read); Gemma Scope on Gemma-2-2B among 12 models | perplexity and synthetic probes only; no languages; features chosen by damage (they coincide with the most active ones, the authors say) |
| **Chang, Zhang, Thomason, Jia, EMNLP 2025** (P11) | cross-model patching of full-precision module outputs into 3-bit AWQ/NF3 models (MLP outputs close most of the gap), English FineWeb | module level, no SAE features, English |
| **Borobia et al., arXiv 2603.25325 (26 Mar 2026, preprint)** (P31; abstract re-read by me) | Gemma-2-2B among three models under magnitude/Wanda pruning; geometric feature survival does not predict causal importance; rare features survive better | pruning, not quantization; no languages. **Precedent for F4: expect that association/geometric survival need not predict function; F4 is a two-sided test** |
| Bhandari and Wade, arXiv 2609.06473 (P28) | steering-vector directions keep cosine 0.945 to 0.998 across INT8/NF4 | English, no SAE, no languages |
| Deng et al. ACL 2025 (P02); Tang et al. ACL 2024; Andrylie et al. arXiv 2507.11230; Chou et al. arXiv 2507.13410 | language-specific features/neurons: directional ablation raises that language's cross-entropy; steering changes output language | full precision only; no task accuracy; no controls matched on activity as designed here |
| Gupte et al. arXiv 2507.15977 (P20) | Gemma Scope on pruned Gemma-2-2B | pruning only; quantization future work |

## What is therefore NOT claimed as new

- Patching full-precision activations into a quantized model to localise damage (Chang et al.) and restoring SAE features to full precision (Sathyanarayanan): the method is known. What is new here is only the **language-group framing with activity-matched controls and per-language recovery**, with weight quantization.
- Ablating language-selective features hurts that language in full precision (Deng, Tang, Andrylie): known; our full-precision part (F1) is a replication-with-controls, not a finding of interest by itself.
- That feature directions persist under quantization (Bhandari and Wade, for steering vectors).

## Candidate contribution (to be earned by the results, not assumed)

1. The causal role of **language** feature groups per quantization level against activity-matched controls (F2).
2. Per-language localisation of quantization damage to those groups versus other features and the SAE error term (F3).
3. Whether the association retention of D-035 predicts functional retention (F4), with a documented pruning precedent for "no".
4. Belebele headroom of a 2B base model across the 27 languages as a by-product.

## Methodological points taken from the check (agent-reported sources; used as design inputs, each to be cited from the register)

- Preserve the SAE error term in feature edits (Marks et al., ICLR 2025; Makelov et al., ICLR 2025: necessity tests keep the reconstruction error): the design already does.
- Report resample and mean ablation next to zero ablation (Heimersheim and Nanda: noising and denoising bring the model less off-distribution). Zero and resample are in the design; **a dev-based mean ablation is added** (design section 5).
- Match control groups on firing rate, activation scale **and decoder norm** (Makelov: high-frequency features overshadow); decoder-norm matching is added as a reported check (design section 4).
- Ablate groups, not single features (splitting, absorption: Chanin et al.; non-canonical units: Leask et al.).
- A difference-in-means language direction (Deng's directional ablation) as a **baseline** that tests whether the SAE group adds anything over a simple direction; added as an optional sensitivity (design section 5).
- bf16 round-off may exceed INT8/INT7 weight noise: the INT16/INT12/INT0 conditions are the floor; INT8/INT7 effects are read against them (design section 2).
- Belebele headroom: no published Gemma-2-2B-pt per-language numbers were found; the IT model's group averages are about 55 (South-East Asian), 47 (African) and 35 (Indic) against chance 25 (Franken-Adapter, arXiv 2502.08037, agent-reported). Indic languages are near the floor, so the coverage rule of the design is essential; small base models may need the cloze format (agent-reported, unverified for Belebele), hence both formats are run.
- Belebele passages come from FLORES-200: handled by the dev/devtest origin split (design section 3).
- Use log-probability of the correct option (not accuracy alone) and paired tests; Holm over the declared family (design section 6).

## Not verified

The Neurocomputing version of Borobia et al. (paywall); models and tasks of Kojima et al. and Kurz et al.; CRANE's tasks; the attribution of an Alignment Forum post on pathological SAE errors; any Gemma-2-2B-pt Belebele number; whether "Wang et al. 2025" (named in the request) exists; the workshop venue of Gemma Scope; whether anything newer than about October 2026 exists.

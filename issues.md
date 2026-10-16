# Code & results review: issues found (documentation only, nothing was fixed)

> **Note (2026-10-16, night).** This review was written against the pipeline that has since been **discarded** (`src/` and the Phase-1 scripts; recoverable from the git tag `legacy-pipeline-2026-10-16`). File and line references below point into that tag, and the Phase-1 result files moved to `reports/legacy_pipeline/`. The findings stay here as evidence; the post-mortem (what was wrong, why results conflicted, whether the direction was worth it) is the last entry of `reports/progress_log.md`.

Reviewed 2026-10-15 against commit `3c17b8e`. Scope: everything in `src/`, `scripts/`, `reports/`, `paper/main.tex`, `README.md`, and the proposal (`anlp_revised.txt`).

**How things were checked**

- Static read of all code.
- Re-derivation of every number in the paper's results tables from the JSON in `reports/` (Fisher p-values, Spearman ρ and so on).
- An independent re-implementation of the token-budget chunking, using the real Gemma-2 tokenizer and the real FLORES-200 dev split. It reproduces the paper's per-language token budgets exactly (20,094 / 20,096 / 20,087 / 20,079 / 20,091 / 20,093 / 20,095 for eng / deu / kor / ind / tam / tel / rus), so the chunking description below is confirmed, not guessed.
- Small numpy simulations (bf16 rounding in the quantizer, the null distribution of S).
- **Not done:** re-running the model, SAE, selectivity or survival. HF downloads ran at ~100 KB/s from this machine, so the 5.2 GB checkpoint was not feasible. Items that need the model are marked *(unverified)*.

Severity: **S1** = a headline claim is contradicted or unsupported; **S2** = weakens validity or deviates from the pre-registered design; **S3** = minor, latent, or hygiene.

---

## 0. TL;DR

1. **The core machinery looks correct and the numbers are not a crash-bug artifact.**
   - The JumpReLU encode, the layer-12 `resid_post` hook, the RTN quantizer, Pearson and classification, and the dev/devtest split all behave as specified.
   - Survival degrades smoothly and monotonically with bit-width (see §2).
   - Every number in the paper's tables traces back to the JSON.
2. **The interpretation in the paper, abstract and README is not supported by the data.** The "RQ2 null, so the gap is upstream in tokenization" story is contradicted by the project's own INT6, INT5 and INT4 results and by a language-level test (A1). The "selective features degrade faster" claim (RQ1) is a 0.1–0.3 pp difference at ceiling (A2). Several paragraphs still carry numbers from the old 1,500-token run.
3. **The instrument was never validated against the paper it claims to replicate (Gate G0 was not run).** Our INT7/INT6 survival is 98%/94% where Duan reports 81%/51%. Duan's released code and table suggest the reference pipeline is the odd one out: no BOS token and a baseline WikiText-2 perplexity of 458 (B1). Either way, the G0 gate as written is not a usable target, and "INT7/INT6 confirmatory" is not justified under the proposal's own instrument-validity rule.
4. **Several design/implementation choices undermine the selective-vs-universal and tier contrasts:**
   - content is not matched across languages (English sees 774 dev sentences, Khmer 127);
   - the bootstrap-stability safeguard was never implemented;
   - the "95th-percentile null" is not what the code computes;
   - feature-level Fisher tests treat 2,342 clustered features as independent;
   - the controls are measured on a different token population than the selective features.

---

## 1. What the results say (20k tokens/language run, the one in the paper tables)

| Question | Result | Reading |
|---|---|---|
| Selectivity landscape | 2,342 of 11,670 supported features (20%) have S(f) ≥ 3. By tier: High 839, Mid 444, Low 1,059 (45%). Only 19 of 27 languages have ≥50. English has 369; Russian has 16. | Plausible, but S ≥ 3 is a weak criterion (C3), and counts are confounded by content coverage (C1) and tokenization. |
| RQ1: selective vs matched universal | INT7 99.87% vs 99.96% (p=0.63). INT6 97.01% vs 97.35% (p=0.53). Only 3 of 2,342 selective features fail at INT7. | At ceiling and indistinguishable. This is not evidence of "selective degrade faster". |
| RQ1, descriptive INT5 / INT4 | INT5: selective 75.3% vs 70.6% (Fisher p=4e-4, selective more robust). INT4: 30.3% vs 35.3% (p=3e-4, selective less robust). Top-5-per-language subset at INT5: 90.4% vs 62.0%. | The sign flips with bit-width. No consistent feature-type effect. |
| RQ2: tier gap (selective features only) | Survival, High / Mid / Low: INT7 100 / 100 / 99.7. INT6 99.05 / 99.3 / 94.4. INT5 81.5 / 88.1 / 65.0. INT4 35.0 / 40.1 / 22.4. Median r is lowest for Low at every bit-width. | A small, consistent, reproducible low-resource penalty. Mid ≥ High, so it is not monotonic. |
| Language-level tier test (what the proposal pre-registered; computed here from `by_language` in the GPU JSON) | Exact permutation, 9 vs 9 languages, one-sided. Per-language median r: p ≈ 0.0002–0.0003 at INT7, INT6, INT5 and INT4. Survived fraction: INT7 p=0.24 (ceiling), INT6 p=0.003, INT5 p=0.0006, INT4 p=0.0008. | The tier effect is real in this data, so the paper's "null" is wrong (A1). |
| What drives the INT6 gap | Yoruba 92.4%, Amharic 87.3%, Lao 90.3%, Hausa 94.2%, Igbo and Sundanese 97.2%. Kannada and Telugu (low tier) are at 100%. | Latin-script low-resource languages are as fragile as Lao and Khmer, which argues against a pure script explanation. Still confounded by tokenization and content (C1, C8). |
| Behavioral link | Spearman ρ(selective drift, ΔL) = 0.495 (p=0.009) at INT7 and 0.484 (p=0.011) at INT6, n=27. Overall-feature drift: ρ ≈ 0.03 and 0.07. | Reproduces. Partial ρ controlling for tier falls to 0.33 and 0.37. INT7 ΔL spans −0.006…+0.012 (9 of 27 are negative), which is noise level. Exploratory only. |

**Good or expected?**

- **Expected:**
  - survival falls monotonically with bit-width;
  - rare, low-resource-language features are more fragile;
  - r ≥ 0.9 is a very low bar at INT7/INT6 for per-channel weight-only RTN, so both arms sit at ceiling.
- **Not "good" for the proposal as written:**
  - the confirmatory INT7/INT6 contrasts have essentially no resolving power;
  - the only informative bit-widths (INT5/INT4) are descriptive-only in the pre-registration;
  - the headline narrative disagrees with the data.

---

## 2. What was checked and looks correct

- `JumpReLUSAE.encode` (`src/sae_models.py:36-40`): `mask = pre_acts > threshold; mask * relu(pre_acts)` matches the Gemma Scope spec. Weight keys and shapes were verified against the real `params.npz` in the progress log (d_model=2304, d_sae=16384).
- Hook (`src/pipeline.py:29-42`): `model.model.layers[12]` output is `resid_post` of block 12, which matches Gemma Scope `layer_12`. Handles both tuple and tensor returns.
- `rtn_fake_quantize` (`src/rtn_quant.py:21-32`): symmetric per-output-channel RTN. Targets are exactly the 7 Linear types × 26 layers (182 modules). Embeddings, norms and lm_head are excluded. Weights are restored from CPU copies.
- `pearson_corr_per_feature`, `classify_survival`, `firing_rate`, `filter_supported_features` and `selectivity_score` are correct as written.
- Selection on `dev` and evaluation on `devtest` are properly separated. The selectivity and survival scripts do not share data.
- Internal consistency of the response curve: `1 − median r` of selective features is 0.0063 / 0.0112 / 0.0241 / 0.0574 / 0.1646 for INT8…INT4. Divided by the relative weight error of ideal RTN on Gaussian weights (0.0083 / 0.0168 / 0.0341 / 0.0704 / 0.151), the ratios are 0.76 / 0.67 / 0.71 / 0.81 / 1.09. That is the expected near-linear scaling for threshold-flip noise, so bit-widths are not mixed up and there is no hidden noise floor.
- Paper tables vs JSON:
  - Fisher p-values reproduce exactly (0.625, 0.534, tier tests).
  - The behavioral Spearman values reproduce (0.495, p=0.0086 by permutation).
  - 0 of 2,342 selective features have a NaN r at every bit-width.

---

## 3. A. Claims that contradict or outrun the data (S1)

**A1. "RQ2 null, so the gap is localized upstream (tokenizer/embeddings)" is contradicted by the project's own results.**
Locations: `paper/main.tex:36` (abstract), `:312`, `:334-337` (discussion), `README.md:42`.
- The paper's own Table (`main.tex:284`) rejects the null for the INT6 tier gap (p=1.1e-8), yet the abstract, discussion and README say the categorical test fails to reject and "proves" the gap is not from selective-feature damage.
- INT5 and INT4 tier gaps are large (65 vs 82; 22 vs 35).
- The language-level test the proposal pre-registered gives p≈0.0003–0.003 where it has any resolution (§1).
- The text says median r "decreases monotonically from High (0.9908) to Mid (0.9921) to Low (0.9850)". It does not (Mid > High).
- The abstract and discussion still quote old-run values: r = 0.9990 / 0.9892, ">91%", "0.9748 vs 0.9958".
- Even if the INT7 test were the only one considered, failing to reject at ceiling is not evidence of absence. The pre-registered TOST/equivalence analysis was never implemented (C4).

**A2. RQ1 "selective features degrade faster in both regimes" is unsupported** (`main.tex:36`, `:297`, `:287`, `README.md:35-36`).
- The gaps are 0.09 pp (INT7) and 0.34 pp (INT6), with p=0.63 and 0.53.
- The sign reverses at INT5 (p=4e-4) and again at INT4, and the top-5 design points the other way at INT5 (90.4% vs 62.0%).
- README numbers (96.30% / 99.15%, n=135 / 117) come from the old run and are not reproducible from anything in the repo (see D4).

**A3. Risk paragraph says 27/27 languages had sufficient features (min 19 Ukrainian, 48 Kannada)** (`main.tex:400`).
- In the 20k run, 19 of 27 languages have ≥50 selective features. The minimum is Russian at 16, Ukrainian is 24 and Kannada is 54 (the quoted values are from the 1.5k run).
- The pre-registered demotion trigger ("fewer than 20 of 27 languages", `anlp_revised.txt` §6) fires under that reading. `progress_log.md` already notes this, but the paper says the opposite.

**A4. Methods and appendix text describe a procedure the code does not run.**
- `main.tex:174` says the null is built from "randomly permuting language labels across sentences over 50 iterations". The code permutes tokens, uses 20 permutations, and uses only the first 6 languages and their first 500 tokens (see C3).
- `main.tex:481` says "the 95th percentile converged stably … by iteration 35" while also saying 20 permutations were run. No code tracks convergence.
- This claim has no code or log behind it. It is a viva risk.

**A5. Behavioral-link paragraph** (`main.tex:324`, `:329`).
- It says the figure plots the "fraction of damaged language-selective features", but Fig. 5 plots `1 − median r`. The damaged fraction is identically 0 at INT7/INT6, so that Spearman is `null` in the JSON.
- It says "confirming that damage to language-specific circuits tracks language-level functional degradation". The actual evidence is ρ≈0.49 with n=27, uncorrected, tier-confounded (partial ρ|tier ≈ 0.33–0.37), from ΔL values at noise level for INT7. It is also never compared against a model with tier or fertility (the OLS promised in the script docstring and the eval-plan table is not implemented) (D6).

**A6. Other stale or incorrect statements in the paper.**
- `main.tex:143`: d = 2,048. Gemma-2-2B has d_model = 2,304 (the progress log verified this).
- `main.tex:233`: "PyTorch 2.5.1+cu121 on Ada cluster (u22)". The final run used torch 2.13 on an RTX PRO 6000 (`requirements.txt`, `progress_log.md`).
- `main.tex:257`: the sanity table says 2,048 WikiText tokens (this matches the JSON). `progress_log.md` says 3,000.
- `main.tex:240`: "reproduced the monotonic degradation behavior reported by Duan". That is true of direction only; the magnitudes are far off (B1).
- `README.md` lists `selectivity_summary_table.csv` and numbers that predate the 20k run (D5).

---

## 4. B. Instrument validity

**B1. Gate G0 (replication of Duan 2026) was never run, and our numbers diverge massively from the reference. The reference pipeline itself looks pathological (S1).**

*Observation.* The reference bands (`anlp_revised.txt` §4.2) are INT7 81.26% ±5 pp and INT6 51.30% ±10 pp. Our demo (`reports/legacy_pipeline/quantization_survival_demo_result.json`) gives 98.28% and 94.31%. The 27-language run gives universal controls at 99.96% and 97.35%.

*What Duan's released code and table show.* I fetched their public artifact (`anonymous.4open.science/r/sae-feature-survival-quantization-0141`, `scripts/03_gemma_phase3_streaming_sweep.py` and `results/tables/phase3_summary_final.csv`) and compared it line by line with ours:

| | Duan (Phase 3, 500k tokens) | Ours (demo, WikiText-test, 2,048 tokens) |
|---|---|---|
| Baseline perplexity (WikiText-2) | **458.5** (loss 6.13) | not measured. English FLORES loss is 3.08 (ppl ≈ 22) |
| BOS token | **none**: `encode(..., add_special_tokens=False)`, raw 256-token windows cut from a concatenated stream and fed to TransformerLens as token tensors | always present (HF tokenizer prepends it) |
| Forward stack | TransformerLens `from_pretrained_no_processing`, bf16 | HF `transformers`, bf16 |
| SAE | `sae_lens` fp32, `canonical` | same checkpoint (verified: `layer_12/width_16k/canonical` → `average_l0_82` in the SAELens registry), but cast to bf16 |
| Quantizer | per-output-channel symmetric RTN, 7 module types × 26 layers; Q/K/V scales are shared across heads because of the TL weight layout (256 scales instead of 2,048) | per-output-channel symmetric RTN, same modules |
| INT8 survival / median r | 99.09% / **0.971** | 99.03% / **0.9999** |
| INT7 survival / median r | 81.26% / 0.950 | 98.28% / 0.9997 |
| INT6 survival / median r | 51.30% / 0.903 | 94.31% / 0.9973 |
| INT5 survival / median r | 27.72% / 0.778 | 82.37% / 0.9849 |
| INT4 survival / median r | 12.40% / 0.451 | 60.40% / 0.9505 |
| Perplexity change vs baseline, INT8 / INT7 / INT6 / INT5 / INT4 | +2.6% / **−5.6%** / +4.0% / +17.7% / +46.1% | not measured |

*Reading.* (Corrected 2026-10-15 after downloading and checking their full released tables; an earlier version of this item overstated the "healthy Gemma" baseline.)
- The quantizer is essentially the same recipe, and it covers exactly the same module set. Their table reports `n_quantized_params = 2,024,275,968`, which equals 26 × (q + k + v + o + gate + up + down) for Gemma-2-2B, the same 182 modules we quantize. The recipe is not the cause. The only coarser part is Q/K/V scale grouping.
- Duan's chunked baseline perplexity is 458.5. Their own sliding-window run on the same model (WikiText-2 test, **with BOS**, 2,048-token windows, HF) gives **46.4**, about 10× lower (`results/tables/gemma_chunked_vs_sliding_ppl_w2048_s512_FULL.csv`). So the chunked, no-BOS, 256-token pipeline is a very different, much worse regime than their own sliding-window pipeline.
- The 46.4 is itself higher than I would expect for Gemma-2-2B on WikiText-2 (my recollection is roughly 10–20; unverified). It may reflect their text joining (empty lines dropped, `"\n\n"` join) or the HF attention backend (soft-capping). It is unexplained, and it must not be taken as a "healthy" reference value.
- The perplexity-vs-bit-width curve is not monotone in either regime. Chunked: +2.6% / **−5.6%** / +4.0% for INT8 / INT7 / INT6. Sliding with BOS: +0.8% / **−6.5%** / **−3.5%**. Real weight quantization should not improve perplexity by 3–6%, and the anomaly survives the BOS regime. So missing BOS is **not** a sufficient explanation for it. Something else (the baseline itself, bf16 noise, or their formatting/backend) also contributes, and it is unresolved.
- Their INT8 median r is 0.971 vs our 0.9999 (about 300× more decorrelation at a nearly lossless width). Survival at r ≥ 0.9 agrees at INT8 (99.09% vs 99.03%) only because the threshold hides it. The pipelines separate from INT7 down.
- The released sliding-window table re-uses the chunked survival numbers verbatim (81.257 / 51.302 / 99.090 in the "sliding" rows). Their `scripts/06_gemma_sliding_window_ppl_check.py` is perplexity-only and contains no SAE or feature code. **No feature-survival measurement under sliding-window or BOS evaluation exists in their release.** The proposal's wording, "even under sliding-window evaluation … only 51.3% of active features survive", pairs a sliding-window perplexity with a chunked-pipeline survival number.
- Their released numbers are internally consistent. Recomputing from `results/per_feature/phase5b_gemma_rtn_int6_vs_pruning_per_feature_overlap.csv` gives exactly their headline INT6 row: 7,144 features, mean r 0.8911, median 0.9027, survived 51.302%, degraded 48.698%, damaged 0.000%. `exp(loss)` equals the reported perplexities, and 499,968 = 1,953 × 256. The released data are not fabricated or mis-summarized, but they describe their pipeline's regime.

*Most plausible explanation of the gap* (evidence-backed, lower confidence than first stated): the G0 bands were produced in a degraded regime (no BOS, 256-token windows, baseline perplexity 458, non-monotone perplexity), where tiny weight changes cause large feature decorrelation. Our pipeline runs with BOS and has English FLORES loss 3.08, which is comparable to their sliding-window loss 3.84, so much higher survival is expected. Contributing differences that were not isolated are the forward stack (TransformerLens vs HF: **correction 2026-10-16**: an earlier version of this sentence said TL computes RMSNorm in bf16; that was wrong, both upcast RMSNorm to fp32. What does differ, found by reading the installed libraries, is that HF's default `sdpa` attention path silently skips Gemma-2's attention-logit soft-capping (cap 50) while TL and HF's `eager` path apply it; our `src/` pipeline runs the sdpa default), the 256 vs 128 window, WikiText vs FLORES, and the fp32 vs bf16 SAE. `reports/audit_our_vs_duan.md` lists every known difference and the experiments (R2, R5) that isolate them.

*Update 2026-10-16, baseline result.* Running their own vendored code on Ada reproduced their published table to within 0.15 pp at every rung (INT7 81.19% vs 81.26%, INT6 51.15% vs 51.30%; baseline perplexity 458.4 vs 458.5; INT7 perplexity -5.61% vs -5.65%; per-feature Spearman 0.9996). So the numbers G0 targeted are reproducible and are not a hardware or one-off artifact; the open question is now why *our* pipeline differs (BOS regime, TransformerLens vs HF, SAE dtype), which the diagnostics layer would answer. See `HANDOFF.md` section 9.

*Update 2026-10-16 (audit of our pipeline vs theirs; `reports/audit_our_vs_duan.md`).* Their regime reproduces inside our HF stack (`T`: INT7 81.41%, INT6 51.30%), and the dominant factor is the BOS token: with one BOS prepended, perplexity is 16.1 instead of ~457 and INT7/INT6 survival is 99.96% / 97.5% (HF `T_mask`; their own TL code with BOS, R2: 99.97% / 97.52%). Their q/k/v scale sharing costs about 8 / 3.5 pp without BOS and nothing measurable with BOS; HF's sdpa path (no attention soft-capping) and a bf16 SAE make no measurable difference. B1's open question is therefore answered for the survival numbers: **their 81% / 51% are a property of feeding windows with no BOS** (D1 in `issues.md` was right to flag BOS handling in our own pipeline, which does include it). What remains open is which regime the study should use, whether their no-BOS regime is signal-limited or noise-amplifying (Gate 1 was only evaluated in the BOS regime), and the INT16/INT12 control there.

*What is still not established.*
- We have never measured our own pipeline's WikiText-2 perplexity, or run it on Duan's exact tokens (WikiText-2 train, 256-token windows, no BOS) with and without BOS. That single experiment would confirm or refute the explanation.
- Under the proposal's rule (§6, "instrument-validity criterion"), failing G0 downgrades survival comparisons to *descriptive*. The paper labels INT7/INT6 as confirmatory.
- If the explanation holds, the G0 bands themselves are not a valid target. The gate should be redefined (e.g. direction plus an FP-vs-FP noise control) instead of "matching 81% / 51%".
- Whatever the cause, our INT7/INT6 are almost lossless for the r ≥ 0.9 criterion, so the confirmatory family has no resolving power (C7).

*How it happened procedurally* (from `reports/progress_log.md`):
- The Duan dry-run was deferred in the log's "Not yet attempted" section as needing "more time than remains before the mid-submission deadline".
- 09-29: the first survival demo came back far above Duan's bands. It was logged as "expected, not a contradiction: small sample", and the paper later says it "reproduced the monotonic degradation".
- `writeoff.md` named G0 as priority #1, but the 10-02 session went straight to the 20k-token re-run, and the log ends with "Gate G0 (Duan replication) was not run".

**B2. No sanity validation of the SAE-on-HF-activations pipeline** *(unverified)*.
- There is no check that mean L0 ≈ 82 and reconstruction/explained variance match the Gemma Scope report on text run through this bf16 HF pipeline.
- There is no explicit `attn_implementation` (`src/pipeline.py:47`). Gemma-2 attention soft-capping is only applied by the eager and some other backends. It is unknown which one ran.
- Why this matters: those two checks are the standard way to rule out a wrong-hook or numerics bug. The sensible selective features found suggest the pipeline works, but it is not demonstrated.

---

## 5. C. Method deviates from the pre-registered design or weakens the contrasts (S2)

**C1. The panel is not content-matched, and token-budget equalization re-introduces a topic confound.**
- Equalizing tokens means each language is truncated at a different sentence. Measured with the Gemma tokenizer on FLORES-dev (997 sentences):

| Language | Dev sentences used | % of dev |
|---|---|---|
| English | 774 | 78% |
| Indonesian | 706 | 71% |
| Mandarin | 703 | 71% |
| Hindi | 417 | 42% |
| Bengali | 286 | 29% |
| Kannada | 231 | 23% |
| Lao | 165 | 17% |
| Khmer | 127 | 13% |

- The devtest used for survival has the same asymmetry.
- Features tied to topics in sentences that only English (or Indonesian/Mandarin) ever sees will look "language-selective". The proposal's content control is broken beyond roughly the first 127 sentences.
- This plausibly inflates English (369 features), and English's median INT6 r is 0.969, as low as the low-resource tier.
- Location: `src/data_utils.py:105-139`.

**C2. The pre-registered bootstrap-stability safeguard was never implemented.**
- `src/survival.py:9-10` lists it in the docstring, but no code does it (peak language must be recovered in ≥80 of 100 resamples of 90% of sentences).
- For features shared by two or three languages (zho/jpn Han characters, Indic or Romance clusters), `argmax` is arbitrary, so tier and language labels are noisy.
- The same applies to rank-normalized S̃: it is computed (`src/survival.py:80-95`) but never used downstream. The proposal's continuous-moderator analysis and threshold-sensitivity sweep are absent.

**C3. "95th-percentile shuffled-null cutoff = 3.000" is not a calibrated null.**
- Where it comes from: `scripts/run_phase1_selectivity_pipeline.py:88-90,113-115`; `src/survival.py:115-151`.
- The null uses only the first 6 languages (eng, deu, spa, fra, por, rus: all high-resource Latin or Cyrillic) and the first 500 tokens of each. S is therefore bounded by 6, but the cutoff is applied to a 27-language S.
- The 500-token sample is the same first ~4 chunks regardless of `--tokens`. This is most likely why the cutoff was exactly 3.000 in both the 1.5k and 20k runs (the "unexplained repeat" in `progress_log.md`); I did not re-run it to confirm.
- The null population is ultra-rare features (mean rate >0.001 over 3,000 tokens means only ≥4 events). Those fall below the ≥50-events support filter later applied to the tested features. The cutoff also passes through `min(raw, max(1.5, 0.75·n_langs))`.
- A proper 27-language null on supported features has a 95th percentile of about 1.34 (simulation: equal true rates in all languages, 20k tokens, support filter applied), and 0% of null features reach S ≥ 3. So the 3.0 cutoff does not come from a null. It is an arbitrary threshold, and it is not what the paper describes.
- What S ≥ 3 means: a feature firing at equal rate in *k* languages has S = 27/k. So any feature present in up to 9 of 27 languages is "language-selective". That is a very loose reading of "language-specific".
- 20% of supported features pass, and the peak-language label for multi-language features is arbitrary (C2).

**C4. Statistical analysis deviates from the pre-registered plan.**
- Location: `scripts/statistical_analysis.py`.
- The pre-registered confirmatory family was 3 tests: INT7 and INT6 selective vs universal, plus a **language-level** directional permutation test with script as covariate. Also pre-registered: a mixed-effects logistic regression with a per-feature random intercept, and TOST equivalence CIs for nulls.
- Implemented instead:
  - feature-level Fisher exact tests (features treated as independent; they are clustered in 27 languages, with many likely near-duplicates, e.g. 237 Yoruba features);
  - no script covariate (`script` is never used in analysis);
  - no mixed model;
  - no TOST;
  - the tier test is run at both INT7 and INT6, giving a 4-test Holm family instead of 3.
- The p=1.1e-8 for the INT6 tier gap is therefore anti-conservative (effective n ≈ languages, not features). It still agrees in direction with the language-level test (A1), so the issue here is validity of the test, not of the conclusion's sign.

**C5. Latent bug: the Holm-Bonferroni implementation has no stop rule** (`scripts/statistical_analysis.py:62-77`).
- Each p is compared to its own `α/(m−rank)` independently. True step-down Holm stops at the first non-rejection.
- Checked directly: `[0.02, 0.0245, 0.9]` returns `[False, True, False]`; Holm should reject none.
- It does not change the current four p-values.

**C6. The selective-vs-universal control is not like-for-like** (`scripts/run_phase1_survival_gpu.py`).
- Selective features' r is computed on their peak language's ~20k tokens (`:168-173`). Matched universals' r is computed on all 27 languages pooled (`:174-175`). Pooled r includes between-language variance in feature rate (which is robust to noise), whereas selective r is within-language only.
- Matching is by firing-rate quintile only (`src/survival.py:154-168`). There is no control for activation magnitude or margin above the JumpReLU threshold. **Duan's own released per-feature data show this is the dominant predictor** (7,144 active Gemma features, INT6): Spearman(r, max activation) = **0.78**, but Spearman(r, firing rate) = **−0.085**. Survival by firing-rate quintile is flat or slightly reversed (56% / 53% / 51% / 53% / 44%). Within *every* firing-rate quintile, features with above-median max activation survive 66–91% vs 21–27% for below-median. The top firing-rate quintile also spans 114× in rate. Rate matching therefore does not control what drives survival, and strongly language-skewed features that fire hard on specific tokens would survive better for that reason alone (consistent with the INT5 and top-5 results, A2). The proposal's premise that "rarity predicts survival, so rate-matching is essential" does not hold marginally in this data.
- Universals are drawn with replacement (`:150-153`, an estimated ~10% duplicate draws).
- 76 selective features (2,342 − 2,266) have no candidate (pooled devtest rate ≤ 0.001) and stay in the selective arm. The arms are unpaired (n = 2,342 vs 2,266).
- `summarize` (`:66-73`) counts NaN r as "not survived" and never as "damaged" (`classify_survival` labels it "undefined"). A feature that dies entirely after quantization would therefore not count as damaged. It is latent here (0 NaNs observed), but it under-reports damage in principle.

**C7. Ceiling and degenerate outcomes.**
- At INT7 and INT6, damaged (r ≤ 0.5) is ~0.0% in both arms. Survival (r ≥ 0.9) is 97–100%. The tests can only see tiny differences.
- The confirmatory widths were chosen from Duan's INT7/INT6, but our quantizer is far gentler (B1).
- The informative bit-widths are INT5 and INT4, which the pre-registration treats as descriptive.

**C8. Tier is confounded with script, tokenization, and language-set composition.**
- `src/data_utils.py:35-66` labels "Dravidian" as a script, but it is a language family (Kannada, Telugu and Tamil use three different scripts). "CJK" lumps Han with kana. The "9 script families" claim is loose.
- The low tier's 1,059 features are dominated by Yoruba, Lao, Khmer and Igbo (69%), and the high tier by English (44%). Pooled-feature tier statistics are driven by a few languages. Any interpretation of "low-resource allocate the largest selective subspace" (45.2%) must account for fertility, content coverage (C1) and token-identity features (e.g. byte-fallback pieces in Khmer/Lao).

---

## 6. D. Implementation details (S2 / S3)

**D1. BOS is counted as an ordinary token and survives into every statistic** (S2; the BOS handling is verified, the effects are *unverified*).
- `encode_texts` and `encode_batched` call `tokenizer(chunk, truncation=True, max_length=128)`, which prepends BOS. Each "128-token" chunk is therefore BOS + 127 content tokens; the 129th token is dropped in 68–100% of chunks. BOS is 0.78% of all tokens in every language (157/20,094).
- Gemma Scope SAEs are trained with BOS excluded (**verified 2026-10-17** in Lieberum et al. 2024, arXiv:2408.05147v2, sections 3.1 and 4: activations from BOS, EOS and padding tokens are not collected and are masked out of the metrics); BOS residuals also have a very large norm (not re-checked).
- Features that fire on BOS get rate ≥ 0.0078 in every language. They enter the "universal" candidate pool and the top rate quintiles, with r dominated by one deterministic spike.
- Locations: `src/pipeline.py:74`, `scripts/run_phase1_survival_gpu.py:43`.

**D2. Decode → re-encode changes the tokens.**
- `build_equalized_token_chunks` cuts at 128 token IDs, decodes to text (`src/data_utils.py:129,137`), and the pipeline re-tokenizes.
- Verified: the re-encoded content differs from the budgeted IDs in 0–32% of chunks: Indonesian 31.8%, Korean 24.8%, Tamil 19.1%, Telugu 18.5%, German 14.0%, most others 0–10%.
- Every chunk also starts mid-sentence without left context. The "exact equal budget" is only approximate (20,079–20,096).

**D3. Everything is bf16; no fp32 reference or noise-floor control.**
- Model weights are bf16. `load_sae` rounds the SAE's fp32 parameters to bf16 (`src/pipeline.py:55-57`). Pre-activation vs threshold comparisons are in bf16.
- The quantizer divides, rounds and multiplies in bf16 (`src/rtn_quant.py:27-32`). In a numpy bf16 emulation, 24% / 32% / 22% of weights land off the ideal RTN grid at INT8 / INT7 / INT6, and the extra error vs the ideal grid is about as large as the INT8 quantization error itself. The total weight error changes by about 7% at INT8, about 2% at INT7, and less beyond that, so this is minor.
- The "FP16 baseline" in docs and code is a bf16 model (Duan's "FP16 baseline" is also bf16, `dtype=torch.bfloat16` on CUDA). `progress_log.md` notes ~0.5% of raw activations differ by >0.01 between batch layouts. No FP-vs-FP control (different batch grouping, or an fp32 baseline) was run to establish the noise floor for r.

**D4. The reported run is not reproducible from the repo as documented.**
- `scripts/run_phase1_selectivity_pipeline.py:55` defaults to `list(PANEL_27.keys())`, which includes the `cmn_Hans` alias. The README commands omit `--languages` and would run 28 "languages" with Mandarin twice. The progress log says this was hit once and worked around by passing 27 codes.
- The README reproduction uses `run_phase1_survival_quintile_matching.py` (CPU, top-5-per-language). The reported numbers come from `run_phase1_survival_gpu.py` (all selective features).
- `.gitignore` excludes `*.npz`, so the `.features.npz` that `--tracked all` needs is not in the repo.
- `phase1_survival_results_27lang.json` is a byte-identical copy of the `_gpu.json` (checked with `cmp`). The old 1,500-token reports were overwritten, so the README's numbers cannot be traced.
- The GPU script hard-codes `device="cuda"` and concatenates all languages' activations on-device. That is ~17.7 GB at 20k tokens/language (`:137`), so it needs a very large GPU.

**D5. Stale and synthetic artifacts remain in the repo and in wandb.**
- `scripts/analyze_selectivity.py` still draws Fig 1, 4 and 5 from `np.random` (lognormal/exponential: `:94,139-141,156-157`).
- It has hard-coded fallbacks: `n_active=10436`, median S=5.2, median firing rate=0.004 (`:68-69,79-80`).
- `reports/legacy_pipeline/selectivity_summary_table.csv` therefore contains those constants and stale 1.5k numbers (eng 229, etc.).
- `figures/fig1_selectivity_distribution.png`, `fig4_selectivity_vs_firing_rate.png` and `fig5_firing_rate_comparison.png` are synthetic. `fig3_selective_pct_by_tier.png` divides by the hard-coded 10,436. They are copied to `paper/figures/`.
- `scripts/log_to_wandb.py:98` uploads every `fig[2-5]_*.png`, so the synthetic ones are logged as if they were results.
- `main.tex` does not include the synthetic ones; it uses the real figures from `generate_mid_submission_figures.py` (`fig2_selectivity_populations.png` and so on). Both sets share the `figN_` prefix, which makes mix-ups easy.

**D6. Minor inconsistencies in definitions and logs.**
- Two incompatible "fertility" definitions:
  - `run_behavioral_loss_link.py:58-67` uses tokens per whitespace word, which gives 11–30 for Chinese, Japanese, Thai, Khmer and Lao (unspaced scripts);
  - the paper's table uses the token ratio vs English.
- The behavioral ΔL (`run_behavioral_loss_link.py`) uses only 1,000 tokens/language (about 8 chunks). Per-token loss is not comparable across languages with very different fertility.
- The `progress_log.md` entry for the demo says 3,000 tokens; the JSON and paper say 2,048.
- `TopKSAE` (`src/sae_models.py:70-99`) was never run against real weights (extension only).

**D7. Process note.**
- `writeoff.md` §0.5 flags that the course forbids AI-generated report text and that parts of the prose "read as possibly AI-generated".
- Related: there are claims in the paper with no code or log behind them (A4's convergence statement).
- This is not a code issue, but both are viva risks. Every number or procedure in the write-up should be traceable to a script or log entry.

---

## 7. Suggested order of attack (not started)

1. Settle the Duan discrepancy (B1): measure our own WikiText-2 perplexity, then run our pipeline on Duan's exact tokens (train split, 256-token windows) with and without BOS, and run Duan's script verbatim. Redefine G0 accordingly. Add the L0 and explained-variance sanity check (B2). Verified venue (Ada u22, 2 × RTX 2080 Ti), run plan and closure protocol: `reports/duan_codebase_audit.md` §4–§7 and `PLAN.md`. The baseline runs their own vendored code (`third_party/duan2026/`, whitelist enforced by tests; `HANDOFF.md`); BOS/INT0 experiments are diagnostics, not replication.
2. Re-derive the confirmatory family at the bit-widths that discriminate (INT6 to INT4), with the language-level permutation test, a script covariate, and TOST. Report the 27 languages as the unit (C4, C7).
3. Fix the content confound by evaluating every language on the *same* sentence set (cap to the shortest-budget common prefix, or accept unequal tokens and account for it) (C1).
4. Implement bootstrap peak-language stability, calibrate the null on the supported population over all 27 languages, and report threshold sensitivity (C2, C3).
5. Make the control like-for-like: evaluate both arms within-language, match on activation magnitude/threshold margin as well as rate (Duan's data show magnitude is the main predictor, C6), sample without replacement, and exclude BOS (C6, D1).
6. Rewrite the abstract, discussion and README to match the numbers actually in `reports/` (A1–A6), and delete or regenerate the synthetic figures and CSV (D5).

---

## 8. External resources audit (models, SAE, data, reference paper)

Everything the pipeline downloads or depends on, and what I found. "Verified" means I checked it against a primary source this session. Otherwise it is a concern from reading the code.

| Resource | How it is used | Findings |
|---|---|---|
| **`unsloth/gemma-2-2b`** (HF model, bf16, 5.2 GB) | Base model for everything. Substituted for the gated `google/gemma-2-2b` (fp32, 10.5 GB) because of disk and gating (`progress_log.md`). | *Verified:* `config.json` matches the official Gemma-2-2B architecture (hidden 2304, 26 layers, 8 heads / 4 KV heads, head_dim 256, attn softcap 50, final softcap 30, sliding window 4096). It is the **pretrained** model, which is the correct pairing for the `-pt-` SAE. *Concerns:* it is a third-party re-upload. It was never byte-compared with the official checkpoint, so "identical" rests on the mirror's claim. The fp32→bf16 rounding is a small extra perturbation (~0.17% relative error vs ~0.8% for INT8). No revision or hash is pinned (`from_pretrained(model_id)`), so the weights could change under us. |
| **Gemma tokenizer** (same mirror, 256k SentencePiece) | Tokenization, budget equalization. | Works as intended. It prepends BOS (D1). Heavy fragmentation for Khmer/Lao/Amharic/Indic scripts (fertility 3–5.5× English) is inherent and is exactly the confound the study discusses, but it also makes "selective features" in those languages likely to be token- or byte-level features. |
| **Gemma Scope SAE** `google/gemma-scope-2b-pt-res`, `layer_12/width_16k/average_l0_82` (JumpReLU) | Fixed measurement basis for all features. | *Verified:* this is the same SAE Duan loads as `layer_12/width_16k/canonical` (SAELens registry maps it to `average_l0_82`), so the SAE is **not** a source of the G0 discrepancy. Param keys and shapes were checked against the real file in the progress log. *Concerns:* (1) it was trained on Gemma's pretraining-style data, which is English-dominant and, as I recall from the Gemma Scope report, with BOS excluded. A 16k dictionary has little capacity for Khmer, Lao and the like, so low-resource "features" may be coarse or token-level, and the SAE's reconstruction quality per language was never measured (B2). All conclusions are about this SAE's dictionary, not about the model's features. (2) One layer, one width, one sparsity level. (3) The loader casts the fp32 parameters to bf16 (D3). (4) No revision pinned. |
| **FLORES-200** (original NLLB tarball, `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) | `dev` for selectivity tagging, `devtest` for survival and loss. | *Verified:* every panel language has 997 dev sentences and the file layout matches the loader. *Concerns:* (1) this is the original FLORES-200, not FLORES+, which corrected translation errors in several languages. Per-language translation quality was not checked. (2) The sources are Wikinews/Wikijunior/Wikivoyage, English-original and translated (translationese), short and out of context. These sentences are public web text, so contamination in Gemma's pretraining is possible. That affects absolute loss values, not the selectivity logic. (3) Under token equalization each language sees a different subset of the sentences (C1), which breaks the "content-matched" property. (4) No checksum on the download. |
| **Joshi et al. (2020) resource taxonomy** | Defines the High/Mid/Low tiers, the main moderator for RQ2. | *Verified against the authors' `lang2tax` file:* the panel's tiers are **not** a clean threshold on Joshi's 0–5 classes. High = class 5 (eng, deu, spa, fra, zho, jpn) plus class 4 (por, rus, kor). Mid = class 4 (hin, tur, vie, pol) plus class 3 (ben, tam, ind, tha, ukr). Low = class 2 (yor, amh, hau, lao) plus class 1 (kan, tel, sun, ibo, khm). Class 4 is in both High and Mid. The proposal says assignments were "cross-checked against NLLB training-corpus mass", but nothing in the repo does that. Tamil is class 3 while Kannada/Telugu are class 1, so even the "Dravidian" group straddles tiers. The tier variable is partly arbitrary, which weakens RQ2 in particular. |
| **WikiText-2** (`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split) | Only the quantization demo (2,048 tokens). | English-only and tiny. Duan used the train split with 500k tokens. Fine as a smoke test, not as a replication corpus. |
| **Duan (2026), arXiv:2606.03002 + anonymous artifact** | Source of the G0 bands and of the "perplexity can miss feature damage" framing. | Single-author preprint. The released code shows the baseline perplexity is 458 with no BOS, and a non-monotone perplexity-vs-bits curve (B1). Treat its absolute survival numbers as specific to that pipeline, not as ground truth. The other 2026 references are listed as unverified in `paper/citations_to_verify.md`. |
| `transformers==5.17.0`, `torch==2.13.0`, etc. (`requirements.txt`) | Runtime stack. | Pinned and consistent with the logs. They disagree with the paper's "PyTorch 2.5.1" (A6). I could not cross-check these versions here. |
| W&B | Logging only. | No scientific effect. It uploads synthetic figures (D5). |
| *Planned, not used:* Llama-3.2-1B + `seonglae` TopK SAE, EleutherAI SAEs | Extension. | Never run. `TopKSAE` is untested (D6). |

General reproducibility gap: none of the three downloaded artifacts (model, SAE, FLORES) is pinned by revision or hash, so a later re-run could silently use different files.

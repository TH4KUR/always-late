# Verification of the language-association-retention plan (D-035)

Date: 2026-10-21. Written by Claude at the user's request ("ensure the plan is factually correct and not already done in other papers, or overexaggerated, before starting implementation and queueing jobs"). Scope: [the design (section 7)](../survival_method_reassessment.md#7-selected-next-study-language-association-retention-d-035), [the implementation plan](../../docs/superpowers/plans/2026-10-21-language-association-retention.md) and the sources they cite ([register](../research_sources.md)). This file is a check, not paper prose. When first written it changed none of Codex's documents; later the same day the user authorised Claude to update her work, and the register and reassessment corrections below, the plan amendments and the status notes were applied (D-036). The corrections stay visible here with their evidence; Codex or the user may overrule them.

## Verdict

1. **The plan's statements about our own code and results are correct.** One technical point needs a change before the GPU smoke (section 3). No factual error was found in any quoted number.
2. **The study is not found in prior work.** No paper found by two independent searches identifies language-selective SAE features before compression and tracks their per-language profile after quantization. The closest works each lack the language dimension, the quantization, or the features (section 2). This is "not found by these searches", not proof of absence, and the plan already declines to claim novelty.
3. **Several statements in the literature register are overstated or incomplete** (section 4). They sit in the supporting narrative, not in the measurement design; none changes what the plan measures. They should be corrected before any of it is quoted in the paper.
4. **Implementation can start** (tests, inventory, collector, analysis code). **No GPU job should be queued** until the smoke gates of Task 5 pass and the storage check in section 3 is done.

## 1. Claims about our code and data, checked against the files

All checks read the committed code and the saved `mask/eq` files; nothing was recomputed that the plan relies on except where stated. The scripts used are not committed (scratchpad); the numbers below are reproducible with `python -I` and the cached run files.

| Plan or register statement | Check | Result |
|---|---|---|
| Per-rung firing/mean/max in `survival.pt` are full-precision values; quantized profiles are not saved | `LangPearson.update` (`phase1_lib.py` lines 74-80), `save_outputs` (`phase1_run.py` lines 186-191) | correct |
| Positive means `z > 0`; S = max r / mean r; nu = mu[l] - mean(mu[others]); support = panel mean >= 1e-5 and >= 50 positives at the peak | `TagStats.update`, `selectivity_S`, `deng_nu`, `supported_mask` | correct |
| Equal-token positions are reconstructible from seed, split offset and budgets | `keep_masks` is seeded by (seed, split_offset, language index) only; `meta.json` records budgets 25,820 (dev) and 27,036 (devtest) | correct for the full 27-language run |
| INT0 reproduces FP exactly | `summary.csv`: INT0 perplexity identical to FP to every printed digit, mean correlation exactly 1.0 | correct (so exact FP/INT0 count equality is a fair gate) |
| Pooled survival 99.97 / 94.94 / 59.00 / 13.75 % and perplexity +0.40 / +2.67 / +11.47 / +103.9 % at INT7/6/5/4 | `reports/phase1/mask/eq/summary.csv` | correct |
| T3 slope -0.004765, Holm p 0.9902, survival-rate-gap p 0.7370, TOST bound 0.02 | `analysis.json` | correct |
| N7 interaction +0.943 (p 0.0013) -> -0.0066 (p 0.982) with magnitude; tier main effect +0.861 (p 3e-06) | `mixed_variants.json` (`full_all`, `log_m_all`) | correct |
| N8 INT6 69.4 / 90.6 / 95.1 %, slope +12.3 pp, Holm 0.0002; N9 slope -0.81 pp (p 0.667); Spearman(scale, survival) +0.966, (tier, scale) +0.833; INT5 residual +5.8 pp (p 0.0022) | `language_level.md` | correct |
| Frequency-matched INT6 gap -7.16 pp; magnitude-matched +4.13 / +4.58 pp; maximum activation 9.77 vs 14.2; Spearman +0.354 | `magnitude.md`, `thresholds.md` | correct |
| 3 h 37 min for the previous mask run | `meta.json` `seconds` = 3.62 h; peak GPU memory 9.6 GB of 11 GB | correct (note the memory headroom) |
| Ada output location and launcher forwarding extra options | `/home2/karma/duan-verify/out/phase1/p1-mask/eq` holds the five files; `run_phase1.sbatch` passes `"$@"` | correct |

Facts the plan should carry (new, from the saved files):

- **Cohort:** 8,086 selective features (every language has at least 73; English 1,719), 2,860 matched controls; selective plus controls equals the whole supported set, 10,946 of 16,384 features. The "frozen union" for sentence-level storage is therefore two thirds of the dictionary, not a small subset.
- **Baseline reproducibility on devtest (eq):** of 8,086 selective features, 6,756 have their dev peak language as the unique devtest peak (89 are tied at the maximum), 7,876 have at least 50 positives at that language, and **6,663 (82.4 %) meet all the plan's baseline-establishment conditions**. The plan's headline fraction will be close to this ceiling for FP, and the lowest per-language shares are por_Latn 39/73, rus_Cyrl 46/83, pol_Latn 87/137. (A first computation of mine gave 31.9 %; it had an indexing error and is discarded; the 82.4 % was recomputed with the correct feature indices; the variant that allows tied peaks gives 6,698.)
- **Deng nu vs S:** the nu-argmax language equals the S peak language for 87.4 % of the selective features, so the two definitions largely agree on the label.

## 2. Is it already done? Prior-art search (2026-10-21)

Two agents searched independently with different angles (about 65 queries, primary pages opened). Their findings were cross-checked against each other; I re-opened one source myself (Verma et al., below) and the FLORES-200 licence. The agents' quotes are not all re-fetched by me; two of my own follow-up fetches were refused by the tool permission layer and I did not work around that.

| Work | Status | What it measures | Overlap with the study |
|---|---|---|---|
| Duan, arXiv 2606.03002 (v2 title "Perplexity Can Miss SAE Feature Damage Under Quantization"; v1 title is the one in our register) | preprint, "submitted to TMLR" in the arXiv comment | the protocol we reuse: frozen Gemma Scope L12 16k SAE, RTN INT8-4, per-feature Pearson survival; WikiText-2 only | **partial: no language, multilingual or FLORES content** (three independent text searches) |
| Gupte, Chhabra, Khalili, arXiv 2507.15977 | preprint, no venue found | Gemma Scope SAEs transferred to Wanda-50% pruned Gemma-2-2B (L12) and GPT-2; reconstruction, SAEBench-style metrics | partial: SAE vs compressed Gemma-2-2B; pruning, English, quantization listed as future work. **Not in our register.** |
| Borobia et al., arXiv 2603.25325; Neurocomputing 704, art. 134826 (Dec 2026 issue) | preprint plus journal article (existence confirmed through Crossref) | feature survival under magnitude/Wanda pruning; Gemma 2 2B, Gemma 3 1B, Llama 3.2 1B | partial: pruning, no language dimension; journal full text not read by anyone |
| Kurz et al., TACL 14:167-192 (2026) | published | LAPE language-specific neurons of Llama-3 8B before/after 50 % SparseGPT pruning, 7 languages | **nearest in spirit**: pre/post-compression language-specific units; but neurons, pruning, aggregate shifts, not SAE features or per-feature peak language |
| Verma et al., ACL 2026 main (arXiv 2604.05090; I read the abstract) | published (arXiv comment) | language-associated units and SAE decompositions under input changes (romanisation, shuffling) | partial: stability of language-associated units under input perturbation, no weight compression |
| Sin Jing Xuan et al., arXiv 2507.18918 | preprint, no venue found | mean Gemma Scope activations by language, 10 languages, translated English phrases | partial: per-language activation scale; no compression, no selectivity |
| Marchisio et al., Findings of EMNLP 2024 | published | behavioural effect of quantization across up to 23 languages | the behavioural tier claim; no features |
| UniComp, arXiv 2602.09130 (EMNLP 2026 main per arXiv) | preprint/accepted per arXiv | compression across languages | **counter-evidence**: "no consistent additional degradation on low-resource languages" for quantization |
| Deng et al., ACL 2025 | published | the language-selective feature definition (nu), ablation | definition only; no compression |

Consequences for the claims:

- **The measurement is not done elsewhere in what we could find.** The new element is the pre/post-quantization language profile of individually selected SAE features across a 27-language panel. That makes it a validation of how survival was measured, as the plan says, and not an established contribution; a paper claim still needs the closest-work check D-029 asks for, with the table above as its start.
- **The behavioural tier effect is contested in the literature** (Marchisio and Marie-Fujita for; UniComp and Mekala et al. against). Our "tier effect" is a measurement on activation correlation, so it should not be introduced as an established fact about quantization.
- **No prior work supports "activation scale explains the tier effect under quantization".** Soualhi's hypothesis points the other way (peripheral languages use lower-magnitude representations that fall under the clipping threshold), and Silva finds larger English outliers. Our N9 is a statistical attenuation, as D-033 already says.

## 3. Plan-level corrections (technical, not literature)

1. **Subset smoke masks differ from the full run by construction.** `keep_masks` seeds by language index and the budget is the minimum over the languages present, so a three-language smoke does not reproduce the full-run positions. The smoke validates the collector, not the frozen positions; cohort mapping must use language codes, never indices (the saved `peak` indices refer to the 27-language order).
2. **Storage.** The Ada home filesystem has 7.9 GB free (85 % used). Sentence-level shards for roughly 10,946 features over 27 x 1,012 sentences and nine conditions can exceed this unless sparse; my planning figure (about 270 MB per condition if roughly 1,000 distinct features fire per sentence) is an assumption to be measured in the smoke, not a fact. Add a disk gate to Task 5: measured bytes per condition times nine, below the free space, before the full job.
3. **GPU memory.** The previous run peaked at 9.6 GB on an 11 GB card; a per-sentence GPU accumulator must fit in the remaining margin or accumulate on the host.
4. **Tie handling.** The frozen dev peak was an `argmax` (first index on ties); the plan's tie sets apply to the evaluation profiles. Record the number of dev ties in the manifest.
5. **Constraints that no longer hold.** The plan's global constraint "No cron, additional chat" reflects the earlier instruction; the user has since asked for a recurring job every 20 minutes (2026-10-21). Codex owns the wording.

## 4. Literature register: statements to correct (applied to the register and the reassessment on 2026-10-21, D-036)

Three audit agents confirmed that titles, venues and years match the register for all 22 items they checked (P01-P19 and S01-S03), and listed these corrections (each with a quoted phrase and a location in their reports; they are agent findings I have not re-fetched one by one):

Re-read by Claude in the primary PDFs after the audit: P14 (the word "confidence" does not occur; "magnitude" occurs only in "orders of magnitude"), P13 (Table 2 values and the caption's own statement that PT-BR has the heavier tail on average and English the highest maximum; English degrades +18 % against +3.9 %), P03 (the Figure 9 caption says "65K for 2B"), and the FLORES-200 licence (CC BY-SA 4.0 on the facebook/flores dataset card). Everything else in this section is as reported by the audit agents.

- **P14 Olmo (Findings NAACL 2025):** overstated. The paper never measures SAE-latent magnitude (its evidence is perturbation norm vs loss in GPT-2) and the word "confidence" does not occur; delete "or confidence" from the reassessment row and restate the magnitude point as the authors' motivation.
- **P13 Silva (PROPOR 2026):** the register is accurate but understates: Table 2 reports higher mean, P95 and P99 activations for PT-BR, English is higher only on the maximum, INT8 is lossless in both languages and English (the high-resource language) degrades more under naive FP8. The paper's mechanism is weakly supported (two unmatched texts, no intervention).
- **P11 Chang (EMNLP 2025, pp. 3410-3429):** "small residual magnitudes predict large errors" is a correlation at example level (r about -0.58 to -0.83, 3-bit AWQ/GPTQ/NF/EfficientQAT, English text, no Gemma, no plain RTN). The patching experiment localises error rather than showing amplification. The paper notes tension with AWQ (P10, which ties large activations to large error, Eq. 2) and finds long-tail examples less affected (Sec 7.2).
- **P15 Huang (EACL 2026):** "low-resource" is not operationalised in the main text, and Gemma-2-2B aggregate accuracy gains are not marginal (GlobalMMLU 0.326 -> 0.359-0.372); the marginal-gain statement holds for individual languages (Yoruba, Amharic). Appendix C has per-language base-model calibration.
- **P16 Ferrando (ICLR 2025):** add Appendix S (correlations between latent activations and token probability are negligible, Gemma 2 2B r = 0.067 / 0.000) and that the unknown-entity evidence is firing frequency, with Gemma Scope on the same model family.
- **P07 Nelson (ICML 2026):** "improved stability" is qualified (one real model, Pythia-160M, TopK; an expressive encoder worsens identifiability there; dictionary identifiability does not imply code identifiability).
- **P06 Leask (ICLR 2025):** the non-atomicity experiment is GPT-2 only; the stitching result in Appendix A.7.1 tests our exact layer-12 16k Gemma Scope SAE (incomplete relative to a 32k SAE).
- **P03 Gemma Scope (BlackboxNLP 2024):** the Europarl result is for a 65k SAE and measured by delta loss; add the open problem about SAE quantization and the negligible bf16 effect (Appendix D.4).
- **P02 Deng (ACL 2025):** ablation is a decoder-direction projection scored by cross-entropy (it never uses the encoder), English is dropped from the analysis, the SAE width is not stated, and the limitations say low-resource languages are not covered. The agent notes the paper evaluates Gemma-2-2B with Gemma Scope on FLORES (10 languages), making it the closest setup precedent.
- **P12 Imai (Stat. Sci. 2010):** "regression adjustment alone is insufficient" is a gloss, not the paper's statement.
- **P01:** record both titles of 2606.03002 and "submitted to TMLR"; **P19** exists as a published article (Neurocomputing 704, article 134826; DOI record from 2026-08-14), so "not verified" is stale, but its full text has not been read by anyone here; **P05** appeared at COLM 2025 per the OpenReview API record (no explicit acceptance decision seen); **P18**'s headline "tax" is a word-fertility ratio and the character fertility used in our covariate is its stated fallback (our `language_level.md` gloss "tokens per character relative to English" is a repo gloss).
- **Added to the register as P20-P29:** Gupte et al. 2507.15977, Kurz et al. (TACL 2026), Verma et al. (ACL 2026), Marchisio et al. (Findings EMNLP 2024), UniComp, Marie and Fujita, Soualhi, Chimoto et al. (EACL 2026), "Steering Under Compression" (arXiv 2609.06473) and "When Pruning Meets Interpretability" (arXiv 2608.25941).

## 5. What was not verified

- The agents' quotations and some page-level claims were not individually re-fetched by me; ICLR oral/poster status, the Neurocomputing full text and several 2026 acceptance statuses (taken from arXiv comments or Crossref) are unverified.
- Absence of the specific study in the literature rests on the listed queries; Semantic Scholar, DBLP and Google Scholar were not accessible to the agents.
- I did not run a mutation or test pass on code here: nothing was implemented in this step.

## Appendix: points carried over from the retired communication folder

On 2026-10-21 the user asked to stop communicating with Codex and to delete `communication/` (a README protocol, three messages, five replies); the folder is recoverable from commit `58c3fff`. Its decisions and corrections already live in the review file, the register, the reassessment and this file. These points were not recorded elsewhere:

1. **Inventory of the saved Phase 1 files** (checked against `survival.pt`, `selectivity.pt` and the Ada listing of 2026-10-18; a bounded inspection, not a search of every cache):

| class | quantity | where |
|---|---|---|
| available | per-language, per-feature Pearson r between FP and quantized activations for every rung (INT8, 7, 6, 5, 4, 16, 12, 0), shape [27, 16384], at every language (not only the peak) | `survival.pt` `corr` |
| available | FP per-language firing rate, mean and maximum activation; per-language token counts; per-language loss sums and token counts for FP and every rung | `survival.pt` |
| available | FP tagging on dev: S(f), peak language, stability counts, support mask, selective flags, matched controls, strata, Deng's nu [27, 16384] (computed and saved, used by no analysis so far) | `selectivity.pt` |
| not available | quantized per-language firing rate, mean, maximum or variance; quantized peak language or S(f); anything per sentence (support, bootstrap and null need sentence-level counts) | would need a new GPU pass |
| derivable on CPU | comparison of the S-selective set with the nu-selective set on the FP tagging (nu argmax equals the S peak for 87.4 % of selective features); correlation at non-peak languages (undefined where a feature never fires) | existing files |

2. **Cost notes.** A quantized-side extension of `LangPearson` is small in code (quantized `(y > 0)` counts, sums and maxima, test first). Re-tagging on quantized features at every rung is much larger (sentence-level int16 counts are about 0.9 GB of host memory per rung for the dev split). The whole S6 run took 3 h 37 min on one 2080 Ti.
3. **Tooling touched by D-028, D-030 and D-031 (unchanged so far).** `reports/phase1_summary.md` (generated by `scripts/g0/phase1_summary.py`) keeps a BOS-free (`none`) column and static text calling it a sensitivity analysis, and its labels use "survival" without the qualification of D-030; a hand-added interpretation notice sits in the static text (the generated block is protected by the consistency check, the notice is not in the script's `STATIC_TEXT`, so `--init` would drop it). `scripts/g0/reproduce_phase1.py` still regenerates the `none` streams from cached historical outputs (no GPU run). `scripts/g0/phase1_anchor.py` is a code-regression check against the vendored smoke run (D-013), not a numerical target from Duan. Options if the user wants a change: relabel the `none` columns as historical and qualify "survival" (edit `STATIC_TEXT` and `LABELS`, tests follow), drop the `none` columns (rewrite the fixture tests), or leave them.
4. **Facts for the design.** No unseen FLORES split remains (the official archive holds only dev, 997 sentences, and devtest, 1,012; both were used). Every committed Phase 1 result file is reproduced by `scripts/g0/reproduce_phase1.py` (89 of 89 identical at 4 BLAS threads, D-026, `reports/phase1_reproduction.md`).

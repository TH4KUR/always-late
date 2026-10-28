# F5 input inventory (Stage 0 of the survival-and-intervention plan)

Written 2026-10-28 15:30 IST by Claude for [the plan of 2026-10-28](../../docs/superpowers/plans/2026-10-28-survival-and-intervention-publication-rescue.md) (Stage 0: "locate the saved feature-level files; list their keys, feature IDs, quantizers, bit widths and sample counts; check whether the same features can be matched across states; record whether the raw files also contain document or example IDs, margins, activation scales and support-overlap measures"). **Only availability and alignment were checked. No predictor-versus-outcome number was computed** (the one numerical check below, diagonal of `S` against `cos`, tests a definition, not a relation). Raw files stay outside git; the paths below are on the local machine.

## 1. Where the files are

| arm | directory (under `/home/karma/Documents/ANLP/project/transfer_runs/func_inputs/`) | states | files |
|---|---|---|---|
| legacy RTN (old study; never rerun) | `legacy_f5_a/`, `legacy_f5_b/` | a: FP, INT16 (numerical-noise reference); b: FP, INT6, INT4 (INT5 on Ada only) | `f5_stage_a.pt`, `f5_state_<S>.pt`, `meta_f5.json` |
| matched RTN | `jobs/rtnm-f5-int64/` (job 44073) | FP, INT6, INT4 | same |
| GPTQ | `jobs/gptq-f5-int64/` (job 44072) | FP, INT6, INT4 | same |

The INT16 floor files exist only in the legacy run a and are shared by all arms (the legacy rounding, D-108). Analysis outputs per arm: `reports/gptq_extension/results/f5/{gptq,rtnm}/` and the committed legacy `reports/functional_validation/results/f5/`. The feature pool and the D-035 / F4 statuses: `reports/functional_validation/f5/f5_sample.json` (4,382 features in the pool; 1,487 were tested, `--per-cell 12 --subset-seed 0`; roles `k64`, `lost@INT4`, `retained@INT4`, `lost@INT5`, `retained@INT5`, `lost@INT6`, `retained@INT6`; a feature can have several roles).

## 2. What each file holds (keys, shapes, as read on 2026-10-28)

**`f5_stage_a.pt`** (a dict; identical in every arm for the keys it shares; the new runs add two keys for the later optional variants, both unused): `feat_ids` (1,487 SAE feature ids), `feat_lang` (language index 0 to 26 of the feature), `write_features` and `write_rows` (1,487 each; the tested features and their rows), `excluded` (1,487 booleans, none true), `n_active` (number of dev sentences in which the feature is active, 142 to 23,859, median 2,392), `n_ctx` (16 for every feature), `contexts` (1,487 x 16 integer sentence indices into the dev set; the 16 sentences of largest full-precision activation), `dev_lang` (26,919 entries: the language of each dev sentence), `sham_gate` (two exact sham checks).

**`f5_state_<S>.pt`** (one per state; `label`, `read`, `write`):
- *read side* (activation agreement between the state and full precision on the SAME positions, **aggregated per feature over all 442,613 dev tokens**): `corr` (token-level Pearson correlation, the survival measure R1), `jaccard` (firing-set overlap), `scale` (ratio of the quantized to the full-precision activation over positions active in both), `n_active_fp`, `n_active_q`, `n_both`. Arrays of length 1,487.
- *write side* (single-feature ablation at the feature's 16 contexts; fingerprint = centred logit change): `S` (1,487 x 1,487, float16: cosine of the FULL-PRECISION fingerprint of feature i with the QUANTIZED fingerprint of feature j; **the fingerprints themselves are not saved**), `cos` (= the diagonal of `S`, the effect similarity W1; checked: the largest difference between `cos` and the diagonal of `S` is 0.000244, the float16 spacing, in all four new state files), `zero_fp` and `zero_q` (no effect in the full-precision or quantized model), `topk_overlap` (promoted-token overlap), `magnitude_ratio` (effect size in the quantized model over that in full precision), `n_pos` and `n_active_q` (ablated positions), `dnll` (mean loss change of the ablation over the context sentences: the effect size in loss units), `retrieval` (`top1`, `mrr`, `ranks`, ties counted against the true feature).

## 3. Join keys and alignment (checked)

- **One row per feature per state.** All arrays of all states of all four directory sets have length 1,487 and are aligned by position to `feat_ids` of the stage-A file. There is no second row for a feature, so "keep all rows of a feature together" holds by construction; the unit is the feature.
- **The stage-A files of the four sets agree:** `feat_ids`, `feat_lang`, `write_features` and `contexts` are equal element by element in legacy a, legacy b, matched RTN and GPTQ. The 1,487 feature ids are distinct (no id appears in two languages). The full-precision state file is bit-for-bit identical in the new jobs and the old jobs (D-108).
- **Features per language** (27 languages): 12 to 75, median 60 (a stratified subset: the sample is not random over the pool, so population statements need the strata or must be avoided).
- **Missing or degenerate values:** no NaN in `corr`, `cos` or `dnll` in any state; `zero_fp` is 0 for all 1,487 features (every feature has a full-precision effect); the legacy INT4 file has one NaN `scale` and one feature with no effect in the quantized model (`zero_q`); the new INT4 files have none; no all-NaN row in `S`.
- **States and quantizers:** FP, INT16 (legacy rounding, reference), INT6 and INT4 for each of legacy, matched RTN, GPTQ. INT5 exists only for the legacy arm (on Ada, not copied); INT8 only in the legacy run a files on Ada (not copied).

## 4. What the plan asks for, and whether it exists

| item | available? | where / limit |
|---|---|---|
| document or example IDs | **write side: yes** (the 16 context sentences of every feature, as dev-set sentence indices; their language is `dev_lang`); **read side: no** (aggregated over all tokens) | Features share sentences and languages, so rows are not independent; a held-out evaluation must keep language (27) as the cluster, as the existing analysis does. Document-level independence cannot be claimed. |
| margins | **no** in the F5 files | The Phase 1 margin audit (`reports/phase1/mask/eq/margin/`) is a separate, language-level analysis; not joined here. |
| activation scale, quantized-to-FP | **yes** | `read.scale` and `write.magnitude_ratio` per state. |
| activation scale of the feature itself in full precision (a baseline) | **not in the F5 files; yes in Phase 1** | `survival.pt` of the equal-budget Phase 1 runs holds `mean_activation`, `max_activation` and `firing_rate` per (language, feature), 27 x 16,384, on FLORES; **identical in legacy, matched RTN and GPTQ** (checked). Joined by (feature id, feature language). A different corpus than the F5 dev tokens: say so if used. |
| support overlap | **yes** | `read.jaccard`, `n_active_fp`, `n_active_q`, `n_both` per state. |
| original (full-precision) effect size | **yes** | `write.dnll` and `write.n_pos` in the FP state file; the row norm of the fingerprint is not saved. |
| per-feature survival from another measurement (a second predictor) | **yes** | Phase 1 `survival.pt` `corr[bits]` per (language, feature) for 8, 7, 6, 5, 4, 16, 12, 0 bits, per arm. Phase 1 uses the FLORES sentences, F5's `read.corr` the dev set. |
| the committed analysis tables | yes | `f5_tables.csv`, `statements.md`, `f5_results.json` (including the Spearman correlation of R1 and W1 per state under `validity`). |

## 5. What this means for Stage 3 (inputs only, no result)

1. The saved files are enough to join R1, W1 and the baselines per feature and state without reconstruction: feature id, language, state and arm are consistent, and all quantities except the fingerprints and the fingerprint norms exist. **The fingerprints are not saved, so W1 cannot be recomputed from raw fingerprints; Stage 3 step 1 reproduces the committed summary tables from the saved per-feature arrays** (the control of the plan's validity gate 1).
2. **Raw sufficient statistics of the read side are not saved** (only the final `corr`, `jaccard`, `scale`); R1 cannot be recomputed either. The reproduction therefore starts at the per-feature values.
3. The external-panel F5 equivalent does not exist (the external panel has the transfer study, not F5): the plan's step 7 (independent validation on the external panel) is **unavailable** for the F5 outcome; mark it so and do not substitute group-level scores.
4. Held-out features: split by language cluster or by feature within language strata; margins unavailable; sentence-level dependence is real on the write side (shared contexts). These are design points for the Stage 3 freeze, to be written before any outcome is read.
5. No result was computed here; the one numerical check (`cos` against the diagonal of `S`) is a definition check.

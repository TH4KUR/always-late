# Study G: graded gain of language-selective features, per language, across the precision ladder

**THE USER'S MAJOR STUDY (instruction of 2026-10-24). AT THE START OF EVERY RUN (every cron firing and every new session): READ THIS PAGE FIRST, check the status table below, and continue the study before any other work.** The cron prompt (`../cron_prompt.md`, priority 0) and `../STATUS.md` (`Immediate`) say the same.

## The request, in the user's words (verbatim, 2026-10-24)

> first measure baseline before and after each quantization (which features contribute to a language, and which are general)
> then turn down those feature's outputs (per language) by a sensible amount (sweep across 3 ratios of de-amplification) and see if performance for that language deteriorates.
> do the same in the other direction, turn those features up by a sensible amount and check. (sweep these across 3 ratios as well)
>
> i am assuming the set of language selective features for language A is different from that of B and C. if that is the case, i want this study to be run separately for each language to see if it actually changes in the same way across all langauges we are looking at.
>
> so we want to run this experiment like this-
>
> for each of the 27 languages
>    at original precision, at INT8, 7, 6, 5 and 4
>       measure whatever we are measuring. do it by amplifying language selective features up by 3 ratios, and down by 3 ratios.
>
> this would help us see if it improves monotonically with the weights of language selective features, if it degrades monotonically across quantizations, and if it is consistent through multiple languages.
>
> first do this study. this is a major part of this project. record this methodology, results very clearly. when the implementation and runs are complete, analyse the results. keep updating documentation, and work on future plans while this study is running on ada. you can queue jobs and work on another thing alongside. you need not wait for everything to finish before moving to the next tasks. just maintain a written record that you need to come back to this task at the start of every run.

## Where everything is

| what | where |
|---|---|
| design, hypotheses and decision rules (written before any number of this study) | [`../../../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md`](../../../docs/superpowers/specs/2026-10-24-graded-gain-study-design.md) |
| implementation plan with checkboxes (the task list to resume from) | [`../../../docs/superpowers/plans/2026-10-24-graded-gain-study.md`](../../../docs/superpowers/plans/2026-10-24-graded-gain-study.md) |
| live state and ordered next steps | `../STATUS.md` (`Immediate`, and the Log lines headed 'Study G') |
| results | this directory (`baseline/`, `calibration/`, `results/`, `figures/` as they appear) |

## Methodology in brief (the full design with every rule: the spec linked above; written before any number of the intervention study)

**Question.** Does the loss on a language's own text depend monotonically on a gain applied to that language's selective features (three ratios down, three up), does that dependence grow monotonically as the weights are quantized, and do the 27 languages agree?

**Objects.** Model Gemma-2-2B, layer-12 residual, the Gemma Scope 16k SAE; weights quantized by RTN per output channel (FP, INT8, INT7, INT6, INT5, INT4), SAE and activations in full precision; BOS masked. For language L: the **group** G_L = the 64 features selective for L on FLORES dev in full precision with the largest dev mean activation (frozen `groups_manifest.json`, the same set in every state); three **matched controls** C_L,r = 64 supported non-selective features matched one-to-one to G_L on dev activation and firing rate (frozen draws).

**Intervention.** At every non-BOS position the SAE activations of a feature set are multiplied by a ratio alpha (`x' = x + (alpha - 1) * sum a_f W_dec[f]`; the SAE error term is kept; alpha = 1 is the identity, alpha = 0 zero-ablation). Not steering: nothing is added where a feature is silent. The same ratio is applied to G_L and to each control.

**Measure.** Next-token loss (nats per token, token-weighted) on the language's own 300 held-out FLORES devtest sentences (the frozen evaluation ids). Harm = loss after the edit minus the unedited loss in the same state (paired per sentence); net harm = group harm minus the mean control harm at the same ratio; total change = edited loss minus the unedited FP loss; optimal gain alpha* = the ratio with the lowest loss. Uncertainty: paired bootstrap over sentences (2,000 draws).

**Ratios.** Chosen from a calibration on the pilot sentences by a rule fixed before it ran (three down ratios at 10, 30 and 60 % of the FP zero-ablation net harm; the three up ratios are their reciprocals); two fine up ratios (1.1, 1.2) are added because the baseline shows a generic 15 % shrinkage of the activations at INT4 (amendment A1). Zero-ablation is kept as the endpoint.

**Hypotheses (decision rules by counts of languages, out of 27).** H-G1 dose, down (net harm strictly ordered, 24 or more); H-G2 down is specific (interval above 0 at the deepest ratio, 24); H-G3 dose, up (group harm increasing, 24; beyond controls, 20); H-G4 precision (harm does not fall along FP to INT4, tau > 0 in 24 and the median not falling by more than one standard error); H-G5a / H-G5b generic and group-specific repair by turning up at INT4 (14 of 27 for the specific one) and alpha* against the inverse activation shrinkage; H-G6 consistency (sign in 24 languages, FP-INT4 rank correlation above 0.5); H-G7 the baseline of selectivity (not met, see G1 below).

**Runs.** Wrapper `scripts/g0/phase1_gain_run.py` around the frozen runner (never edited); one Ada job per state, 38 configurations per language (about 3 GPU hours per state); analysis `phase1_gain_analysis.py`, figures `phase1_gain_figures.py`; every module has tests written first and a mutation check.

**Status of the evidence.** Exploratory: the devtest sentences were examined in earlier studies (D-039). One model, one layer, one SAE, RTN only, loss not task accuracy.

## Status (update at every step; the date is the last update)

| step | state | note |
|---|---|---|
| G0 design, hypotheses, ratios rule | DONE 2026-10-24 07:03 | spec and plan committed before any number; D-070 |
| G1 baseline before and after each quantization (which features are selective for each language, which are general) | DONE 2026-10-24 07:12 | CPU, D-035 profiles; module `scripts/g0/phase1_gain_baseline.py` (36 tests, 25 mutants killed); results below and in `baseline/`; H-G7 not met |
| G2 wrapper runner (frozen runner untouched), tests first | DONE 2026-10-24 07:26 | `scripts/g0/phase1_gain_run.py` (25 tests, 27 mutants killed), `run_gain.sbatch`; smoke job 36827 (3 languages, 40 sentences, FP and INT4) completed, gates passed, 51 configurations, about 1,500 token passes per second |
| G3 ratio calibration on the pilot split | DONE 2026-10-24 09:12 | FP job 36834 COMPLETED; the rule gave down 0.3536, 0.1768, 0.0884 and up 2.8284, 5.6569, 11.3137 (z = 0.2445); D-071; `calibration/`; the INT4 calibration is deferred as an informational check (it cannot change the ratios) |
| G4 main runs: 27 languages x FP, INT8, 7, 6, 5, 4 x 3 down + 3 up ratios (+ 2 fine), group and matched controls | DONE 2026-10-25 13:06: all six states COMPLETED and validated (999 configurations each, 300 sentences, RTX 2080 Ti) | jobs 36925, 36926, 37693, 37694, 37947, 37948; checksums in `artifact_checksums.md` |
| G5 analysis (monotone in the ratio, monotone across precisions, consistent across languages) | DONE 2026-10-25 13:06: final analysis of the six states (D-078) | H-G1, H-G2, H-G3, H-G6 met; H-G4 narrowly not met; H-G5a, H-G5b, alpha* correlation not met; `results/` |
| G6 figures and the written results | DONE for the main study 2026-10-25 13:06 | `figures/` (four figures, light and dark, looked at); results section below; extensions pending |
| Extension B: state-defined groups (the features that contribute at each precision, selected per precision on the 712 devtest sentences outside the evaluation) | DESIGNED 2026-10-24 10:32 (spec section 10, D-073, Q27); **2026-10-25: manifests done and H-Gb1 MET (D-081); GPU jobs `gain-sd-4` 38816 running, `gain-sd-FP` 38817, `gain-sd-6` 38818, `gain-sd-FPref-4` 38869, `gain-sd-FPref-6` 38870 queued; the analysis module `phase1_gain_extb.py` is built and tested (26 tests, 23 mutants killed); H-Gb2 to H-Gb4 wait for the jobs** (original status: code and runs after the six main jobs) | CPU route from the D-035 per-sentence sidecar; states FP (reference), INT6, INT4; hypotheses H-Gb1 to H-Gb4 |
| Extension C: the repair near ratio 1 with a larger sample (562 unused reserve sentences, 5 control draws, ratios 1.1, 1.2, 1.4, 1.7; INT4 and FP, then INT6 and INT5) | **DONE 2026-10-25 16:40: H-Gc1 MET (through ratio 1.1 only), H-Gc2 MET, H-Gc3 MET by the rule (D-082; results below)**; DESIGNED 2026-10-24 20:32 (spec section 11, rules H-Gc1 to H-Gc4 fixed before any run; plan task list C1 to C3) | motivated by the first look and the reliability audit (D-074, D-075) |
| Extension D: does the optimal gain of a language at INT4 follow the shrinkage of its group's activation? (rank correlation of alpha* with 1 / g_ratio over 27 languages on the extension C data; post hoc, CPU only, no job) | DESIGNED 2026-10-25 17:20 (spec section 12, rules R-D1 to R-D4 fixed before any number, D-083; plan task list D1 to D3) | motivated by D-082 (minimum at 1.1 to 1.2 that contains 1 / 0.852 = 1.17) and by the not-met alpha* correlation of the main study (D-078) |

## G1: baseline before and after each quantization (2026-10-24 07:12; descriptive; D-035 devtest profiles, equal-budget stream, 27,036 tokens per language)

Full tables: [`baseline/report.md`](baseline/report.md) (per state, and per language for FP and INT4), `baseline/baseline.csv` (162 rows: language x state), `baseline/baseline.json`. Inputs (sha256 in the report): the D-035 `profiles.pt` and the Phase 1 dev `selectivity.pt`. Medians over the 27 languages:

| state | features selective for the language (state view) | of its FP-dev cohort still selective | new to the language | group K64 peak retained | group activation ratio to FP | whole-language activation ratio | control-pool ratio | share of the language's activation mass carried by the group |
|---|---|---|---|---|---|---|---|---|
| FP | 258 | 0.797 | 80 | 0.922 | 1.000 | 1.000 | 1.000 | 0.048 |
| INT8 | 260 | 0.796 | 79 | 0.922 | 1.000 | 0.997 | 0.998 | 0.048 |
| INT7 | 259 | 0.787 | 81 | 0.906 | 0.998 | 1.000 | 1.001 | 0.048 |
| INT6 | 263 | 0.777 | 84 | 0.922 | 0.995 | 0.994 | 0.992 | 0.047 |
| INT5 | 283 | 0.722 | 101 | 0.875 | 0.990 | 1.015 | 1.010 | 0.046 |
| INT4 | 250 | 0.561 | 120 | 0.766 | **0.852** | **0.843** | **0.853** | 0.051 |

- **Which features contribute to a language.** The group K64 carries a median 4.8 % of its language's activation mass (the 27 values range from about 2 % to 23 %: yor_Latn 23.3 %, lao_Laoo 16.9 %, amh_Ethi 11.7 %); the whole selective set and the matched-control pool ("general" features, supported and not selective) carry most of the rest (the pool alone about 41 %).
- **What quantization does to selectivity.** The state-view selective set of a language keeps 80 % of the dev cohort at FP (the devtest reproduction ceiling of D-035), 78 % at INT6, 72 % at INT5, **56 % at INT4**, while features new to the language rise from 80 to 120; the share of the group's 64 members whose peak is still the language falls from 0.92 to 0.77 at INT4 (26 of 27 languages decline).
- **The activation shrinkage at INT4 is generic.** Group, whole language and control pool all shrink by about 15 % at INT4 (0.852, 0.843, 0.853); the group relative to the whole language is 1.004 (range 0.889 to 1.109). At INT8 to INT5 nothing moves by more than 1.5 %.
- **H-G7 not met** (its rule: the group's mass share and the group's retained peak both fall along the ladder, in at least 24 of 27 languages at INT4): the mass share does not fall (11 of 27 languages decline); the retained peak falls in 26 of 27 languages but its median is not non-increasing (0.922 at INT8, 0.906 at INT7, 0.922 at INT6: one member of 64).
- **Consequence for the intervention study (recorded before any intervention number): spec section 9 (amendment A1).** If the INT4 loss is partly this generic shrinkage, scaling any 64 features up by about 1.17 should lower the INT4 loss (generic repair), and the group is special only if its net harm against matched controls is below zero. Two fine up ratios (1.1, 1.2) are added to the runs for that test.
- Limits: devtest profiles, no intervals; the state view is looser than the cohort's dev definition (no bootstrap stability); a descriptive baseline, not a test of function.

## First look at the main runs (2026-10-24 19:50; PREVIEW on FP and INT4 only; D-074; `preview_FP_INT4/`)

The two finished main jobs were analysed with the unchanged analysis module before the other four states exist. **Verdicts of the pre-registered rules on these two states:** H-G1 (dose, down) met, 27 of 27 languages strictly ordered, median net harm 0.0223 / 0.0902 / 0.1687 / 0.2624 nats per token at ratios 0.354 / 0.177 / 0.088 / zero-ablation; H-G2 (down is specific) met, 26 of 27; H-G3 (dose, up) met (increasing in 27, net harm above 0 at 11.3 in 24); H-G4 (precision) met on the two states (27 of 27, to be judged again on all six); H-G6 (consistency) met (positive net harm in 27 of 27 languages at both states, language rank correlation FP against INT4 0.936). **Not met:** H-G5a (generic repair), H-G5b (12 of 27 languages against 14 needed), the alpha* correlation (-0.034). **What the numbers say besides the verdicts:** the zero-ablation harm is 1.6 times larger at INT4 (0.4316 against 0.2624); every down ratio costs more at INT4; at INT4 scaling the group by 1.1 or 1.2 slightly LOWERS the loss (median group harm -0.0014 and -0.0017, controls about 0), alpha* is above 1 in 21 of 27 languages (7 of 27 at FP) with the median at the fine-grid edge 1.2: suggestive of a group-specific repair of about a thousandth of a nat per token, below the pre-registered threshold. Figures of the real data (light theme, preview) were drawn and looked at; three layout defects found on real data (heavy-tailed values flattening the panels, ticks at ratios that are not in the study, a scatter stretched by one language) were fixed with tests. Final analysis and figures follow when all six states are in.

## Sample-size floor and reliability audit (2026-10-24 20:12; user instruction; D-075)

The user asked that the sample never be reduced so far that the empirical results become unreliable. **Floor for every run of Study G: 27 languages, 6 states, 300 held-out sentences per language, 3 matched-control draws; follow-ups only add.** The main study is at the floor. The reliability audit of the first two states (new section 'Reliability of the sample' of every report and `reliability.csv`): the dose-response down ratios, zero-ablation and large up ratios are well measured (23 to 27 of 27 languages exceed twice the combined error of sentences and control draws); the repair near ratio 1 is not (effects of 1 to 3 standard errors, 14 to 16 of 27 languages clear), so the preview's 'not met' for H-G5b is partly a power statement; any follow-up on the repair uses MORE sample (the 562 unused reserve sentences, 5 control draws), plan Extension C.

## Second preview, four states (2026-10-24 23:53; D-076; `preview_FP_INT6_INT5_INT4/`)

FP, INT6, INT5 and INT4 analysed with the unchanged rules (INT7 and INT8 still running). H-G1, H-G2, H-G3 and H-G6 met; **H-G4 not met, marginally**: the per-language ordering is perfect (27 of 27) but the median group harm at the deepest down ratio falls from FP (0.1939) to INT6 (0.1861) by 0.0078, more than the larger bootstrap error (0.0069); INT5 0.1940, INT4 0.3807. The precision dependence is flat until INT5 and appears at INT4; the sign change of the fine-ratio effect (net harm -0.0010 at x1.1, alpha* above 1 in 21 of 27 languages) also appears only at INT4, not at INT6 and INT5 (+0.0008, +0.0006; alpha* above 1 in 7 of 27). The rule is not changed; the final report prints the same clause with two errors as a labelled post hoc sensitivity.

## RESULTS of the main study (2026-10-25 13:06; final analysis of six states; D-078; `results/report.md`; exploratory)

**Pre-registered rules (design section 5, amendment A1):** H-G1 dose, down: **met** (27 of 27 languages strictly ordered, median net harm 0.0223 / 0.0902 / 0.1687 / 0.2624 at ratios 0.354 / 0.177 / 0.088 / zero-ablation at FP). H-G2 down is specific: **met** (26 of 27). H-G3 dose, up: **met** (27 and 24 of 27). H-G6 consistency: **met** (27 of 27 positive at every state; rank correlation FP against INT4 0.936). H-G4 precision: **not met, narrowly** (per-language ordering perfect, 27 of 27; the median at the deepest ratio is 0.1939 at FP, 0.1820 at INT8, 0.1861, 0.1861, 0.1940 at INT7, INT6, INT5 and 0.3807 at INT4; a fall of 0.0119 from FP to INT8 exceeds one bootstrap error; with two errors, labelled post hoc, it is met). H-G5a, H-G5b (12 of 27 against 14), alpha* correlation: **not met**. H-G7 (baseline): not met.

**What the data say (medians over 27 languages, nats per token):**
1. *Graded and specific.* The harm of turning a language's group down rises step by step to zero-ablation and, turned up, to 1.2 nats per token at 11.3 times; every step is larger than for the matched controls.
2. *A threshold at INT4.* The dose-response is flat from FP to INT5 (net harm at ratio 0.0884: 0.1687, 0.1690, 0.1632, 0.1658, 0.1795) and 1.9 times larger at INT4 (0.3233); zero-ablation: 0.2624, 0.2681, 0.2664, 0.2709, 0.2808, 0.4316. The same threshold shows in the activation shrinkage of the baseline (1 to 1.5 % until INT5, 15 % at INT4).
3. *Turning up costs less at INT4.* Net harm at 11.3: 0.8095 at INT4 against 1.2128 at FP.
4. *A repair signal at INT4 only.* At INT4 the group scaled by 1.1 or 1.2 slightly lowers the loss (median group harm -0.0014 and -0.0017; net -0.0010 and -0.0013; controls about 0) and alpha* is above 1 in 21 of 27 languages (5 to 7 at the other states; median 1.2, the edge of the fine grid); at FP to INT5 the fine ratios cost a little. It is about a thousandth of a nat per token and under-powered at this sample (14 to 16 of 27 languages clear twice the error): **a suggestion, not a result**; extension C tests it with more sentences and draws and rules fixed beforehand.
5. *The up-side dose axis is not language-independent:* the net harm at 11.3 as a multiple of the language's zero-ablation harm ranges from -8.7 to 72.9.

Figures (`figures/`, light and dark, svg and png): language panels (FP), macro dose-response by state, language x state heat maps, the repair figure. **Limits:** exploratory (devtest examined before); one model, layer, SAE and quantizer (RTN); loss on the language's own text; the frozen K64 full-precision group; intervals conditional on three control draws (reliability table: 23 to 27 of 27 languages clear twice the combined error for the main dose-response); all six states on an RTX 2080 Ti.

**Extension C status:** `gainc-FP`, `gainc-6`, `gainc-5` (2080 Ti) done and fetched; `gainc-4` ran on an RTX 4070 and is a hardware sensitivity run only (D-079); `gainc-4b` 38720 (2080 Ti) is queued for the INT4 verdicts.

**Update 2026-10-25 16:40: EXTENSION C RESULTS (D-082; `extension_c/`, figure `figures_extension_c/`).** `gainc-4b` 38720 (RTX 2080 Ti, 3 h 33 min, 675 configurations, gates passed, complete) is validated and fetched; all four states ran on a 2080 Ti on 562 reserve sentences per language with five matched-control draws. The rules of spec section 11 were applied as written:

| rule | result | verdict |
|---|---|---|
| H-Gc1 repair at INT4, ratio 1.1 | median net harm -0.00135 [-0.00197, -0.00032] (97.5 % interval over languages), 20 of 27 languages negative (sign test p = 0.0096) | met |
| H-Gc1 repair at INT4, ratio 1.2 | median -0.00169 [-0.00311, +0.00027], 18 of 27 negative (p = 0.061) | not met at this ratio; the hypothesis asks for either ratio: **met through 1.1 only** |
| H-Gc2 INT4 minus FP | -0.0024 [-0.0042, -0.0012] at 1.1; -0.0041 [-0.0071, -0.0025] at 1.2 | met: the same turn-up costs at FP (+0.0006, +0.0014; 7 and 4 of 27 languages negative) and lowers the loss at INT4 |
| H-Gc3 group-specific | matched-control median -0.00015 [-0.00036, +0.00004] at 1.1; +0.00013 [-0.00036, +0.00082] at 1.2 (neither interval entirely below 0) | met by the rule, narrowly: at 1.1 the controls' point estimate is about a tenth of the group's, a small generic part cannot be excluded |

At INT4 the median net harm is -0.0013, -0.0017, -0.0007, +0.0046 at 1.1, 1.2, 1.4, 1.7: **a shallow minimum near 1.1 to 1.2 that is gone by 1.4 and a cost at 1.7**; FP, INT6, INT5 show a cost at every ratio (medians +0.0004 to +0.0014 at 1.1 and 1.2). Three independent samples agree on the sign at INT4 (pilot D-080, evaluation D-078, these reserve sentences). **Size and limits:** a thousandth to two thousandths of a nat per token (about 0.05 % of the loss); at INT4 and the two primary ratios 17 to 23 of 27 languages are individually clear of twice their error (13 to 26 over all states and ratios); unit of the rules = the language; exploratory (devtest, one model, layer, SAE and quantizer); the rules were fixed before these data, the hypothesis came from the main-study data. Post hoc, labelled: the inverse of the INT4 group-activation ratio (0.852, baseline G1) is 1.17, inside the minimum. **Reading the figure:** the interval endpoints are percentiles of a bootstrap of 27 values and move with the order of the languages in the resample (the analysis fixes the roster order and seed 0 and writes the numbers to `bands.csv`; the figure reads them and never recomputes); the verdict at 1.1 does not depend on it (20 negative of 27 gives P(median of a resample >= 0) = 0.0035, far from 0.0125), neither does the failure at 1.2 (18 negative); the lower end of the 1.2 interval moved from -0.0031 to -0.0060 in a recomputation with another order.



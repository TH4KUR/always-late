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

## Status (update at every step; the date is the last update)

| step | state | note |
|---|---|---|
| G0 design, hypotheses, ratios rule | DONE 2026-10-24 07:03 | spec and plan committed before any number; D-070 |
| G1 baseline before and after each quantization (which features are selective for each language, which are general) | DONE 2026-10-24 07:12 | CPU, D-035 profiles; module `scripts/g0/phase1_gain_baseline.py` (36 tests, 25 mutants killed); results below and in `baseline/`; H-G7 not met |
| G2 wrapper runner (frozen runner untouched), tests first | not started | |
| G3 ratio calibration on the pilot split | not started | |
| G4 main runs: 27 languages x FP, INT8, 7, 6, 5, 4 x 3 down + 3 up ratios, group and matched controls | not started | |
| G5 analysis (monotone in the ratio, monotone across precisions, consistent across languages) | not started | |
| G6 figures and the written results | not started | |

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


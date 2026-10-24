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
| G1 baseline before and after each quantization (which features are selective for each language, which are general) | not started | CPU, from the D-035 `profiles.pt` (all 16,384 features, every condition) |
| G2 wrapper runner (frozen runner untouched), tests first | not started | |
| G3 ratio calibration on the pilot split | not started | |
| G4 main runs: 27 languages x FP, INT8, 7, 6, 5, 4 x 3 down + 3 up ratios, group and matched controls | not started | |
| G5 analysis (monotone in the ratio, monotone across precisions, consistent across languages) | not started | |
| G6 figures and the written results | not started | |

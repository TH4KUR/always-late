# Gemma-2-9B against the project's findings: three quantizer arms (confirmation of the updated project)

The predictions, thresholds and verdict functions were frozen before any 9B number existed (`reports/functional_validation/gemma9b/amendment_A1_three_arm.json`, text in `docs/superpowers/specs/2026-10-28-gemma9b-three-arm-confirmation-design.md` section 5). All numbers come from the frozen Study G and Phase 1 code. Reference values are the committed Gemma-2-2B outputs (the same statistics).

## Tier 1, Study G (loss caused by removing a language's features): **consistent**

27 languages tested (group of at least 16 members), 300 held-out sentences per language, three matched-control draws.

| prediction | rule (fixed beforehand) | Gemma-2-9B | Gemma-2-2B | result |
|---|---|---|---|---|
| Q0 existence and grading under the better quantizers | P1 at FP and H-G1 at INT4 in the matched-RTN and GPTQ arms | rtn-matched: P1 met, H-G1 INT4 met; gptq: P1 met, H-G1 INT4 met | met in both arms | **met** |
| Q1 the INT4 amplification is smaller with GPTQ | r(GPTQ) <= r(matched), r(GPTQ) <= r(legacy) - 0.3, GPTQ below legacy in at least 20 languages; evaluable only if the legacy arm meets P3b | r legacy 1.689, matched 1.490, GPTQ 1.165; GPTQ below legacy in 26 languages; legacy arm meets P3b: yes (median ratio 1.689, larger at INT4 in 26 of 27, needs 20 and a ratio above 1.3) | r 1.911 / 1.575 / 1.239; 25 of 27 | **met** |
| Q2 6 bits are the same in every arm | macro retention at INT6 within [0.85, 1.15] in all three arms | legacy 1.006, rtn-matched 1.040, gptq 1.015 | flat in all arms | **met** |
| Q3 repair hint (reported) | H-G5b per arm; languages whose best factor at INT4 is above 1 | legacy: H-G5b not met (8 of 27), best factor above 1 in 14; rtn-matched: H-G5b not met (3 of 27), best factor above 1 in 8; gptq: H-G5b not met (2 of 27), best factor above 1 in 4 | H-G5b not met under matched RTN and GPTQ; best factor above 1 in 21, 13, 7 of 27 languages (legacy, matched, GPTQ) | reported |

Legacy-freeze predictions of the legacy arm (P1 to P3b, verdict function of 2026-10-26): P1 met, P2a met, P2b met, P3b met, P3a met; label **consistent**.

## Tier 2, Phase 1 survival and loss: **consistent**

27 languages, equal-budget stream. Means over languages in the order legacy RTN / matched RTN / GPTQ.

| prediction | rule (fixed beforehand) | Gemma-2-9B | Gemma-2-2B | result |
|---|---|---|---|---|
| S1 survival ordering | GPTQ > matched > legacy at INT6 and INT4, GPTQ above legacy in at least 24 languages | INT6 95.1 / 96.7 / 99.0 %, INT4 22.3 / 32.9 / 46.6 %; GPTQ above legacy in 27 / 27 languages | INT6 85.0 / 89.7 / 97.5 %, INT4 12.7 / 18.6 / 30.5 % | **met** |
| S2 loss ordering | GPTQ < matched < legacy at INT6 and INT4, GPTQ below legacy in at least 24 languages | INT6 0.0157 / 0.0092 / 0.0034, INT4 0.410 / 0.287 / 0.088 nats/token | INT6 0.0254 / 0.0197 / 0.0058, INT4 0.590 / 0.385 / 0.130 | **met** |
| S3 the tier gap at 6 bits | high minus low mean survival at INT6 smaller under GPTQ than under legacy | gap 9.1 / 6.3 / 1.7 points | gap 25.7 / 19.5 / 4.8 points | **met** |
| S4 the original tests T1 to T3 (reported) | the Holm-corrected p per arm | legacy: below 0.05: T2_int6; rtn-matched: below 0.05: T2_int6; gptq: null; arms agree: False | legacy null, matched RTN below 0.05 (T1, T2), GPTQ null | reported |

## Reading and limits

- Both verdicts are separate and are not merged. consistent / partly consistent / different are defined in the amendment; a result that is not consistent is a valid result.
- This is a scale replication within one architecture (Gemma 2) and one SAE family (Gemma Scope, layer 20 of 42, width 16k): one model, one layer, one dictionary, weights-only rounding. It says nothing about other architectures, layers or SAEs.
- Contrasts are per arm; no multiplicity rule covers three arms. The Gemma-2-2B legacy numbers are the committed ones; the 2B matched-RTN and GPTQ arms are exploratory (D-101 to D-110).
- Hardware: one RTX PRO 6000 (Blackwell) with a different torch build from the 2B runs. The artifact hashes of the 9B weight sets are recorded on this GPU type; equality on other hardware is not claimed. The 2B bridge (bit-identical full-precision numbers) does not exist for 9B.
- GPTQ saw Wikipedia text in the 27 languages and RTN saw none: a GPTQ-versus-RTN gap mixes error compensation and calibration; matched RTN separates the grid and granularity change only.

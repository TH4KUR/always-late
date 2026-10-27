# Verified future-work anchors and novelty boundaries

> **Historical assessment / design-stage record.** The [26 October post-D-094 novelty assessment](../novelty_assessment_2026-10-26.md) supersedes novelty verdicts and forward priorities below. The broad claims have direct precedents; an exact protocol match was not found, but novelty is not verified. Keep this source/search history; use current results and the recovery plan for decisions.

Date: 26 October 2026. Targeted primary-source review, not an exhaustive systematic review.
This register supports the main plan and two conditional branches. A future-work statement
establishes motivation, not novelty or a likelihood of a positive result.

## Explicit author statements

| Anchor | Verified location and short quotation | What it supports | What it does not support |
|---|---|---|---|
| F1: Duan, Perplexity Can Miss SAE Feature Damage Under Quantization, arXiv:2606.03002v2 | PDF p.11, conclusion and section 7: “view causal validation as the next step” | Main direction: test whether activation changes alter interventions under quantization; Branch B: behavioral transfer. | Our particular factorial method is our proposal, not an author-specified experiment. This is a preprint/under-review manuscript, not verified as accepted TMLR. |
| F2: Hoang et al., FEGA, arXiv:2607.24645v1 | PDF p.22, section 8: “Applying FEGA at successive layers could reveal whether and where a localized feature effect disperses” | Branch A: locate propagation and dispersion of an intervention effect; study how compression changes that propagation. | The extension to compression is our hypothesis. FEGA already measures output effect clouds. Preprint; no accepted venue verified. |
| F3: Chang et al., EMNLP 2025 | PDF p.9, section 9: “We invite future work beyond intrinsic analysis to extrinsic, downstream tasks.” | Main plan's requirement for behavioral validation; Branch B's output-language versus comprehension distinction. | This broad invitation does not establish novelty of a particular multilingual experiment. This is a verified peer-reviewed EMNLP paper. |
| F4: FEGA, section 8, PDF pp.21–22 | Paraphrase: authors propose naturalistic generation beyond structured tasks, and warn about retained-context selection. | Branch B: held-out generation and per-context checks; main plan avoids averaging away context failures. | Not permission to claim that effect cancellation or naturalistic evaluation is new. |

Primary links:
- https://arxiv.org/abs/2606.03002
- https://arxiv.org/abs/2607.24645
- https://aclanthology.org/2025.emnlp-main.168/
- https://aclanthology.org/2025.emnlp-main.168.pdf

Each quotation is deliberately short. PDF pages refer to the versions queried above.
Duan's p.11 discussion concerns simulated RTN and does not itself validate interventions.
FEGA's exact layerwise proposal is about propagation of one intervention, not training new SAEs at every layer.

## Closest work already done: requirements on our novelty claim

- **Borobia revision:** https://github.com/hecboar/sae-pruning-paper .
  Revised README directly inspected in the previous research round. Includes frozen-dictionary
  controls, corrected BOS/dtype evaluation, activity-controlled error-preserving local ablations,
  and INT8/INT4 pilots. The older arXiv-only “pruning not quantization” distinction is inadequate.
- **FEGA:** downstream effect-cloud geometry is existing methodology. Use and cite it.
- **Chang:** noisy activation versus weight restoration and MLP-gate patching are existing.
  A finding that activations matter more than weights is not new by itself.
- **Strength Controls Change How Interventions Compare Across Sparse Autoencoder Codes**,
  https://arxiv.org/abs/2610.26489 (19 October preprint): norm and output-calibrated controls;
  also has shared coefficients learned from other inputs. Earlier suggestions that held-out
  shared calibration alone would distinguish us were too broad. Do not claim that as new.
  Scope is varying sparse codes on fixed models, unlike our proposed compressed-model factorial.
- **Representation Redistribution in Sparse Autoencoders**,
  https://arxiv.org/abs/2610.21240 (15 October preprint): recurring representation change
  versus signed fixed-readout effects and failure; held-out evaluation. No “geometry ≠ function”
  novelty claim.
- **Behavioral Compositionality of Sparse Autoencoder Representations**,
  https://arxiv.org/abs/2610.21261 (15 October preprint): interactions beyond single-feature
  effects, random-rotation controls, held-out coalition behavior. Pairwise interaction modeling
  is not an untouched backup direction. Extracted experimental text states main pipelines use
  unquantized evaluation; an extension to compression still needs a distinct result.
- **What Do Sparse Autoencoder Feature Discovery Scores Identify?**,
  https://arxiv.org/abs/2610.15850 (11 October preprint): observational scores fail to identify
  intervention attribution; gradient attribution predicts local interventions.
  A first-order gradient predictor must be a baseline, not our proposed invention.
- **When Are Sparse Feature Interventions Actually Localized?**,
  https://arxiv.org/abs/2607.10226 (July preprint): matched surface/basis/strength and coherence
  controls. One model is loaded in 4-bit; therefore do not say it has no quantized models.
  Its focus is sparse-versus-dense safety localization, not our factorial comparison.
- **Recurrence Is Not Enough: Causally Validating Multilingual SAE Translation Features**,
  https://arxiv.org/abs/2609.04808: translation-feature causal evaluation already exists.
  This round retrieved it as a novelty threat; previous local register has abstract-level checks.
  Before Branch B, read its full current text and compare protocol/results.
- **Deng et al., ACL 2025**, https://aclanthology.org/2025.acl-long.229/:
  full text queried; language identification, continuation steering and feature gating already
  done in FP. No exact matching future-work passage was verified here; not used as a future-work
  anchor. Branch B cannot claim those tasks as new.

The main candidate delta is a replicated, controlled explanation of NATIVE intervention-transfer
failure under weight compression, plus a diagnostic that adds held-out predictive value over
strong baselines. The crossed design is a means, not a sufficient contribution.

## Search record

Existing 26 October memo contains the initial search strings and eight-source comparison.
This follow-up queried primary PDF passages for F1–F4 and the newest direct competitors.
One alphaXiv discovery call targeted crossed state/weight experiments and layerwise effect
dispersion. Follow-up web searches:
1. "SAE" "quantization" "causal" "transfer" intervention
2. "SAE" "quantization" "future work" steering
3. "quantization" "feature" "intervention" "context" sparse autoencoder downstream weights
4. "sparse autoencoder" "quantization" "layerwise" effects
5. "language" "steering" "quantization" SAE transfer
6. "FEGA" "quantization"

Discovery returned F1/F2, 2607.10226, 2606.18322, 2610.17898, 2609.12591, 2607.20596,
2609.22782, 2610.21261, 2609.39243, 2609.04808 and 2610.15850. Full relevant pages of
2607.10226, 2610.21261 and 2610.15850 were read; the others are leads, not verified exclusions.
Search results for theses, author projects, and generic reviews were not treated as peer-reviewed
future-work evidence. ACL Anthology venue pages were checked for Chang and Deng.

## Honest novelty status

No identical completed factorial study was found in the searched sources. An absence claim
cannot be proved by this search. A project explicitly listed as future work has necessarily
been talked about; the relevant distinction is an unresolved question versus a completed answer.
If a full-text competitor matches the final actual contribution, revise the claim or stop.
A credible preprint plus a peer-reviewed broad motivation supports investigation; it does
not establish an A-tier result. Recheck updated versions before submission.


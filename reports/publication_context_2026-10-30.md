# Context for the completed-study paper

Written 30 October 2026. This is a **targeted narrative literature check**, not an exhaustive novelty or systematic-review claim. The question is how to interpret scoped SAE intervention evidence in light of current work on feature nonuniqueness, functional evaluation and quantization.

## Search and selection

Primary sources: ACL Anthology, official ICLR proceedings, PMLR, author/institution publication records, and arXiv for explicitly labeled preprints. Search window: work available by 30 October 2026; English. Include directly relevant primary methodological or empirical work; exclude aggregators, forum posts and unverified publication claims. Existing project source-register entries supplied candidate works, and the official records below were opened to verify attribution and the claims used.

| Date | Search/source | Query or record | Use |
|---|---|---|---|
| 29–30 October | ACL Anthology | `2025.acl-long.229 language-specific features sparse autoencoders`; `2024.blackboxnlp-1.19` | Language-associated SAE interventions and Gemma Scope provenance. |
| 29–30 October | Official ICLR proceedings | `Sparse Autoencoders Do Not Find Canonical Units of Analysis`; `Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control` | Distinguish noncanonical coordinates from useful task-grounded causal experiments. |
| 29–30 October | PMLR | `SAEBench comprehensive benchmark sparse autoencoders`; `Sparse but Wrong Incorrect L0` | Proxy/functional divergence and dependence of disentanglement on sparsity. |
| 29–30 October | arXiv, OpenReview, ETH/ISTA author records | `2606.03002`; `GPTQ Accurate Post-Training Quantization Frantar Ashkboos Hoefler Alistarh` | Quantization-feature audit precedent and calibrated quantization method. |

This was a small purposive set of known works, with records deduplicated by title/identifier; no screened-population count or exhaustive novelty conclusion is inferred from it.

## What the literature supports

- **Language-associated SAE coordinates can be interventionally useful.** [Deng et al., ACL 2025](https://aclanthology.org/2025.acl-long.229/) identify and manipulate language-associated features. Our contribution is not their existence; it is the comparison of preservation measurements and interventions under controlled quantization.
- **A dictionary need not be canonical to support a useful experiment.** [Leask et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/84ca3f2d9d9bfca13f69b48ea63eb4a5-Abstract-Conference.html) demonstrate incompleteness and non-atomicity and explicitly retain practical SAE usefulness. Their evidence qualifies ontological claims, not every specified decoder-direction contrast.
- **Functional validation can be task grounded without recovering exact atoms.** [Makelov et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/53356aebeea8ffd40a8ac3bb66243162-Abstract-Conference.html) evaluate disentanglement and control on a task, agnostic to exact ground-truth feature recovery. This motivates reporting the actual intervention and endpoint.
- **Metrics measure different things.** [SAEBench, ICML 2025](https://proceedings.mlr.press/v267/karvonen25a.html) finds that proxy gains do not reliably imply better practical performance. Our activation/effect comparison addresses a related distinction under compression; it does not show all proxies are useless.
- **SAE training choices can change semantic granularity.** [Chanin and Garriga-Alonso, ICML 2026](https://proceedings.mlr.press/v306/chanin26a.html) study sparsity-dependent disentanglement. We do not claim our frozen dictionary solves that general problem; keeping it fixed defines a reproducible paired measurement.
- **Behavioral parity does not certify frozen-feature fidelity.** [Duan, arXiv 2606.03002v2](https://arxiv.org/abs/2606.03002) motivates activation auditing. Our added question is how activation agreement relates to controlled effect preservation; the BOS audit restricts numerical comparisons across input regimes.
- **GPTQ changes the quantization procedure using calibration.** [Frantar et al., arXiv 2210.17323](https://arxiv.org/abs/2210.17323) describes the method; [the authors' publication record](https://spcl.ethz.ch/Publications/index.php?pub=477) confirms its ICLR 2023 appearance. The paper bibliography cites the identifiable GPTQ preprint title; the conference paper uses the name OPTQ. Our comparison does not isolate compensation from calibration-domain effects.

## Reporting consequence

The paper presents a controlled empirical/measurement contribution. The fixed SAE basis is a deliberate operational choice: paired interventions remain interpretable while semantic completeness and external validity stay bounded. Shared difficulties in mechanistic interpretability justify explicit estimands and controls; they are not an excuse for stronger claims. Conversely, broader architecture/dictionary sweeps would expand external validity rather than retroactively create validity for every existing observation.

Publication suitability depends on editorial review, clarity, relevance and comparison with the submission venue's current literature. This revision claims a coherent empirical contribution, not acceptance or first-ever priority. The canonical questions and answers are in [`FINAL_RESEARCH_QUESTIONS.md`](FINAL_RESEARCH_QUESTIONS.md).

"""Study G, extension B (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 10; D-073, Q27): groups of language-selective features selected IN each precision state, on devtest sentences
that the evaluation never sees.

Input: the per-sentence shards of the D-035 association sidecar (`<root>/<stream>/sentences/<state>/<language>.pt`: `sentence_ids` 0..S-1, `n_tokens` of the stream per sentence, the supported feature ids `feature_ids`,
CSR rows `row_ptr`, `col`, `counts` int16 positives, `sums` float32). The profile of a state sums the rows of every sentence that is NOT in the evaluation set. Selection at that state uses the state view of
`phase1_gain_baseline.state_classes` (support rule, unique peak, S above the dev null cutoff); a language's group is its first K `own` features ranked by mean activation on the language at that state (ties by feature
id, as `phase1_func_groups.group_members`); its controls are drawn with the frozen matching (`phase1_func_groups.draw_controls`) from the state's supported non-selective pool on the state's own statistics.
The output is a groups manifest in the frozen format, so the unchanged runner runs it with `--manifest`. Only features present in the shards (supported on FP dev) can be selected."""
import hashlib
from pathlib import Path

import numpy as np
import torch

import phase1_func_groups as fg
import phase1_gain_baseline as gb


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shards(root, stream, state, languages):
    """({language: shard dict}, {language: sha256 of the shard file}) from `<root>/<stream>/sentences/<state>/<language>.pt`."""
    shards, shas = {}, {}
    for l in languages:
        path = Path(root) / stream / "sentences" / state / f"{l}.pt"
        shas[l] = _sha256(path)                                 # raises FileNotFoundError for a missing shard
        shards[l] = torch.load(path, weights_only=False)
    return shards, shas


def profile_from_shards(shards, exclude_ids, n_features):
    """{"n_tokens" int64 [L], "positive_counts" int64 [L, F], "activation_sums" float64 [L, F]} summed over the sentences that are not in `exclude_ids`, in the order of `shards`. Refuses shards whose
    sentence ids are not 0..S-1, whose feature ids differ between languages, or an excluded id outside the sentences."""
    langs = list(shards)
    ref_feat = shards[langs[0]]["feature_ids"].numpy().astype(np.int64)
    n_tok = np.zeros(len(langs), dtype=np.int64)
    counts = np.zeros((len(langs), n_features), dtype=np.int64)
    sums = np.zeros((len(langs), n_features), dtype=np.float64)
    for i, l in enumerate(langs):
        sh = shards[l]
        sid = sh["sentence_ids"].numpy().astype(np.int64)
        n_sent = sid.shape[0]
        if not np.array_equal(sid, np.arange(n_sent)):
            raise ValueError(f"{l}: the sentence ids are not 0..{n_sent - 1} in ascending order")
        feat = sh["feature_ids"].numpy().astype(np.int64)
        if not np.array_equal(feat, ref_feat):
            raise ValueError(f"{l}: the feature ids differ from those of {langs[0]}")
        bad = [e for e in exclude_ids if not 0 <= e < n_sent]
        if bad:
            raise ValueError(f"evaluation ids {bad} are outside the {n_sent} sentences of {l}")
        keep = np.ones(n_sent, dtype=bool); keep[list(exclude_ids)] = False
        row_ptr = sh["row_ptr"].numpy().astype(np.int64); col = sh["col"].numpy().astype(np.int64)
        rows = np.repeat(np.arange(n_sent), np.diff(row_ptr))
        use = keep[rows]
        np.add.at(counts[i], feat[col[use]], sh["counts"].numpy().astype(np.int64)[use])
        np.add.at(sums[i], feat[col[use]], sh["sums"].numpy().astype(np.float64)[use])
        n_tok[i] = int(sh["n_tokens"].numpy().astype(np.int64)[keep].sum())
    return {"n_tokens": n_tok, "positive_counts": counts, "activation_sums": sums}


def state_selectivity(profile, cutoff, languages):
    """The fields of the frozen dev view (rates, mean_activation, peak, selective, supported, languages) as tensors, from a profile and the dev null cutoff, using the state view of the baseline module."""
    c, n, s = profile["positive_counts"], profile["n_tokens"], profile["activation_sums"]
    cls = gb.state_classes(c, n, cutoff)
    return {"rates": torch.tensor(c / n[:, None], dtype=torch.float32), "mean_activation": torch.tensor(s / n[:, None], dtype=torch.float32), "peak": torch.tensor(cls["peak"], dtype=torch.long),
            "selective": torch.tensor(cls["selective"]), "supported": torch.tensor(cls["supported"]), "languages": list(languages)}


def build_state_manifest(sel, K=64, R=3, seed=0, source_sha256=None):
    """Groups manifest in the frozen format for one state: per language the ranked `own` features (`all`), the group size and R matched control draws of size K. A language with fewer than K own
    features is an error (a smaller group would change the dose silently)."""
    langs = list(sel["languages"])
    label = f"K{K}"
    man = {"schema": 1, "source": {"sha256": source_sha256}, "languages": langs, "Ks": [K], "Rs": [R], "seed": seed, "groups": {}, "controls": {}, "sizes": {}}
    for l, name in enumerate(langs):
        members = fg.group_members(sel, l)
        if members.numel() < K:
            raise ValueError(f"{name}: only {int(members.numel())} features are selective for it in this state, fewer than the group size {K}")
        grp = fg.top_k(members, K)
        d = fg.draw_controls(sel, l, grp, R, seed)
        man["groups"][name] = {"all": members.tolist()}
        man["controls"][name] = {label: d["sets"], "gap": {label: d["gap"]}}
        man["sizes"][name] = {label: int(grp.numel())}
    return man


def overlap_table(manifests, label, ref="FP"):
    """Rows (language, state, shared, jaccard, size) comparing the group (first `sizes[label]` members) of every state with that of the reference state."""
    base = manifests[ref]
    rows = []
    for state, man in manifests.items():
        if state == ref:
            continue
        if list(man["languages"]) != list(base["languages"]):
            raise ValueError(f"{state}: the languages differ from those of {ref}")
        for l in base["languages"]:
            a = set(base["groups"][l]["all"][:base["sizes"][l][label]]); b = set(man["groups"][l]["all"][:man["sizes"][l][label]])
            rows.append({"language": l, "state": state, "ref": ref, "size": len(b), "shared": len(a & b), "jaccard": len(a & b) / len(a | b)})
    return rows

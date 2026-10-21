"""Feature groups and activity-matched control draws for the functional-validation study (D-039/D-040; design section 4). Pure torch/numpy.

Only DEV quantities of the frozen D-035 cohort are read (original `mask/eq/selectivity.pt`): the selective features and their dev peak language, the dev firing rate and the dev mean
activation. No devtest outcome and no quantized-model information enters a group or a control (dev_view drops every other field). Groups are the selective features of a language
ranked by their dev mean activation on that language; controls are drawn from the supported NON-selective pool (the only available pool) by greedy nearest-neighbour matching, one
control per group member without replacement, on the language's own dev (log mean activation, log firing rate) in the L1 sense, because the size of an ablation effect scales with the
activation that is removed. Members are matched from the most active down; each draw picks uniformly among the M nearest still-available pool features, so that the R draws differ.
(Amended 2026-10-21 before any outcome existed: equal-count rate-quintile x magnitude-tercile cells left the group up to 4.6 times more active than its controls in the top cell.)"""
import hashlib
import json
from pathlib import Path

import torch

DEV_FIELDS = ("rates", "mean_activation", "peak", "selective", "supported", "languages")


def dev_view(sel):
    """The dev-only fields of a loaded selectivity dictionary (everything else is dropped)."""
    missing = [k for k in DEV_FIELDS if k not in sel]
    if missing:
        raise ValueError(f"selectivity file lacks {missing}")
    return {k: sel[k] for k in DEV_FIELDS}


def group_members(sel, l):
    """Selective features whose dev peak language is l, ranked by dev mean activation on l (descending, ties by feature id)."""
    ids = ((sel["peak"] == l) & sel["selective"]).nonzero().squeeze(1)
    mass = sel["mean_activation"][l, ids].double()
    order = sorted(range(ids.numel()), key=lambda i: (-float(mass[i]), int(ids[i])))
    return ids[torch.tensor(order, dtype=torch.long)] if order else ids


def top_k(members, K):
    """The first K members (the whole group when K is None or larger than the group): groups for different K are nested."""
    return members if K is None else members[:K]


def pool_ids(sel):
    """Supported features that are not selective: the only control pool of the cohort."""
    return (sel["supported"] & ~sel["selective"]).nonzero().squeeze(1)


def _seed(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def draw_controls(sel, l, group, R, seed, M=3):
    """R control sets (sorted id lists, each with exactly len(group) features) for language l and a group: members in descending dev mass (ties by id), each matched to an unused pool
    feature chosen uniformly among the M nearest still-available ones by |dlog mu| + |dlog rate| on language l's dev statistics; sets use independent seeds (language, size, r).
    Also returns `gap`, the mean |dlog mu| between members and their controls per set (a matching diagnostic)."""
    if M < 1:
        raise ValueError("M must be at least 1")
    pool = pool_ids(sel)
    group = torch.as_tensor(group, dtype=torch.long)
    if pool.numel() < group.numel():
        raise ValueError(f"the control pool has {pool.numel()} features, fewer than the group's {group.numel()}")
    mu = torch.log(sel["mean_activation"][l].double().clamp_min(1e-9))
    lr = torch.log(sel["rates"][l].double().clamp_min(1e-9))
    pm, pr = mu[pool], lr[pool]
    order = sorted(group.tolist(), key=lambda f: (-float(sel["mean_activation"][l, f]), f))
    sets, gap = [], []
    for r in range(R):
        g = torch.Generator().manual_seed(_seed(seed, l, group.numel(), r))
        avail = torch.ones(pool.numel(), dtype=torch.bool)
        chosen, gaps = [], []
        for f in order:
            d = torch.where(avail, (pm - mu[f]).abs() + (pr - lr[f]).abs(), torch.full_like(pm, float("inf")))
            near = torch.sort(d, stable=True).indices[:min(M, int(avail.sum()))]
            j = int(near[int(torch.randint(0, near.numel(), (1,), generator=g))])
            avail[j] = False
            chosen.append(int(pool[j])); gaps.append(float((pm[j] - mu[f]).abs()))
        sets.append(sorted(chosen)); gap.append(sum(gaps) / len(gaps) if gaps else 0.0)
    return {"sets": sets, "gap": gap}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(K):
    return "all" if K is None else f"K{K}"


def build_manifest(selectivity_path, Ks=(16, 64, None), Rs=(5, 5, 3), seed=0):
    """Frozen groups and control draws for all languages from a selectivity file (dev fields only). JSON-serialisable."""
    if len(Ks) != len(Rs):
        raise ValueError("Ks and Rs must have the same length")
    sel = dev_view(torch.load(selectivity_path, weights_only=False))
    langs = list(sel["languages"])
    man = {"schema": 1, "source": {"sha256": _sha256(selectivity_path)}, "languages": langs, "Ks": list(Ks), "Rs": list(Rs), "seed": seed, "groups": {}, "controls": {}, "sizes": {}}
    for l, name in enumerate(langs):
        members = group_members(sel, l)
        man["groups"][name] = {"all": members.tolist()}
        man["controls"][name] = {"gap": {}}
        man["sizes"][name] = {}
        for K, R in zip(Ks, Rs):
            grp = top_k(members, K)
            d = draw_controls(sel, l, grp, R, seed)
            man["controls"][name][_label(K)] = d["sets"]
            man["controls"][name]["gap"][_label(K)] = d["gap"]
            man["sizes"][name][_label(K)] = int(grp.numel())
    return man


def verify_manifest(man, selectivity_path):
    """ValueError unless the manifest was built from this exact source file and equals a fresh deterministic rebuild."""
    if man["source"]["sha256"] != _sha256(selectivity_path):
        raise ValueError("the selectivity file differs from the one the manifest was built from")
    fresh = build_manifest(selectivity_path, tuple(man["Ks"]), tuple(man["Rs"]), man["seed"])
    if json.loads(json.dumps(fresh)) != json.loads(json.dumps(man)):
        raise ValueError("the manifest differs from a fresh rebuild")
    return True


def matching_table(sel, man):
    """One row per language and group size: how well the controls match the group on language l's own dev statistics (means over the R control sets)."""
    rows = []
    for l, name in enumerate(man["languages"]):
        members = man["groups"][name]["all"]
        for K in man["Ks"]:
            lab = _label(K)
            grp = torch.tensor(members if K is None else members[:K], dtype=torch.long)
            sets = torch.tensor(man["controls"][name][lab], dtype=torch.long)
            rate, act = sel["rates"][l], sel["mean_activation"][l]
            g_act, c_act = float(act[grp].mean()), float(act[sets].mean())
            rows.append({"language": name, "size": lab, "n": int(grp.numel()), "R": int(sets.shape[0]),
                         "group_mean_rate": float(rate[grp].mean()), "control_mean_rate": float(rate[sets].mean()),
                         "group_mean_activation": g_act, "control_mean_activation": c_act, "activation_ratio": g_act / c_act if c_act else float("inf"),
                         "mean_log_mu_gap": sum(man["controls"][name]["gap"][lab]) / len(man["controls"][name]["gap"][lab])})
    return rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build, verify or report the frozen feature groups and control draws of the functional study.")
    ap.add_argument("command", choices=["build", "verify", "report"])
    ap.add_argument("--selectivity", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--report-out")
    a = ap.parse_args(argv)
    if a.command == "build":
        man = build_manifest(a.selectivity, seed=a.seed)
        Path(a.manifest).write_text(json.dumps(man, separators=(",", ":")) + "\n")
        print("wrote", a.manifest, {n: man["sizes"][n] for n in man["languages"][:3]}, "...")
    elif a.command == "verify":
        verify_manifest(json.loads(Path(a.manifest).read_text()), a.selectivity)
        print("ok: the manifest equals a fresh rebuild from", a.selectivity)
    else:
        man = json.loads(Path(a.manifest).read_text()); verify_manifest(man, a.selectivity)
        rows = matching_table(dev_view(torch.load(a.selectivity, weights_only=False)), man)
        lines = ["# Matching of control groups to selective groups (dev statistics of the target language)", "",
                 "Generated by `scripts/g0/phase1_func_groups.py report`. Means over the R control sets; `activation` is the dev mean activation (zeros included) on the language, `rate` the dev firing rate; `gap` is the mean absolute difference of log mean activation between a member and its control.", "",
                 "| language | size | n | R | group rate | control rate | group activation | control activation | ratio | log gap |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['language']} | {r['size']} | {r['n']} | {r['R']} | {r['group_mean_rate']:.4f} | {r['control_mean_rate']:.4f} | {r['group_mean_activation']:.4f} | {r['control_mean_activation']:.4f} | {r['activation_ratio']:.2f} | {r['mean_log_mu_gap']:.3f} |")
        text = "\n".join(lines) + "\n"
        if a.report_out:
            Path(a.report_out).write_text(text)
        print(text)


if __name__ == "__main__":
    main()

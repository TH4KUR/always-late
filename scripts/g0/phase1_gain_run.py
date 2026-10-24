"""Study G, task G2 (design: docs/superpowers/specs/2026-10-24-graded-gain-study-design.md section 4): the graded-gain runs through the FROZEN functional-validation runner.

`phase1_func_run.py` is frozen (its hash is tested against the T-LM freeze) and cannot make configurations that scale matched controls. This module adds them without touching the file: it registers a
preset (families: zero-ablation of the group and of the controls, sham, the group scaled by every ratio, the same through the frozen `scale` family) in the runner's `PRESETS` and replaces the runner's
module-level `build_configs` by a wrapper that builds the frozen families and appends the scaled controls. The preset NAME carries the ratios (exact `repr` values), the number of control draws and
the group size label, so the metadata and the result files of every run say which ratios they hold. Names of the configurations (frozen scheme `<family>[draw]_<label>@<language>`):
  scale{alpha}_K64@code        the group scaled by alpha (frozen `scale` family)          cscale{alpha}.{r}_K64@code   control draw r scaled by alpha (this module)
  zero_K64@code, ctrl{r}_K64@code, sham_K64@code    as in the frozen presets (zero-ablation endpoint, zeroed controls, exactness check)
Every configuration runs on its target language's own text only (use `--panel` with no languages: no off-target text)."""
import math

import torch

import phase1_func_run as fr

GAIN_KEY = "gain_controls"
PRESET_PREFIX = "gain"
N_FEATURES = 16384
_FROZEN_BUILD = fr.build_configs


def validate_ratios(ratios):
    """The ratios as a list of floats: finite, positive, not 1 (the identity is the unedited baseline), distinct, and distinguishable in the configuration names (`:g` formatting)."""
    out = [float(r) for r in ratios]
    if not out:
        raise ValueError("at least one ratio is needed")
    if any(not math.isfinite(r) or r <= 0 for r in out):
        raise ValueError(f"ratios must be finite and positive: {out}")
    if any(r == 1.0 for r in out):
        raise ValueError("a ratio of 1 is the unedited baseline, not a configuration")
    if len({f"{r:g}" for r in out}) != len(out):
        raise ValueError(f"ratios must differ in the names of the configurations (6 significant digits): {out}")
    return out


def preset_name(ratios, n_controls, label, zero=True):
    """`gain|<ratios, exact repr>|c<n controls>|<label>` (plus `|nz` when the run has no zero-ablation and no zeroed controls): the preset name of a run (stored in the metadata and in every result file)."""
    ratios = validate_ratios(ratios)
    base = f"{PRESET_PREFIX}|{','.join(repr(r) for r in ratios)}|c{int(n_controls)}|{label}"
    return base if zero else base + "|nz"


def parse_preset_full(name):
    """Inverse of `preset_name`: (ratios, n_controls, label, zero); ValueError for any other preset name."""
    parts = str(name).split("|")
    if len(parts) not in (4, 5) or parts[0] != PRESET_PREFIX or not parts[2].startswith("c") or (len(parts) == 5 and parts[4] != "nz"):
        raise ValueError(f"{name!r} is not a gain preset")
    return [float(x) for x in parts[1].split(",")], int(parts[2][1:]), parts[3], len(parts) == 4


def parse_preset(name):
    """(ratios, n_controls, label) of a preset name (see `parse_preset_full` for the zero-ablation flag)."""
    return parse_preset_full(name)[:3]


def control_scale_configs(man, run_langs, targets, ratios, label, n_controls):
    """The scaled matched controls: for every target language, every one of the first `n_controls` draws of the manifest and every ratio, a `scale` configuration on that language's text only."""
    index = {c: i for i, c in enumerate(run_langs)}
    missing = [t for t in targets if t not in index or t not in man["controls"]]
    if missing:
        raise ValueError(f"targets {missing} are not in the run or in the manifest")
    cfgs = []
    for code in targets:
        draws = man["controls"][code][label][:n_controls]
        if len(draws) < n_controls:
            raise ValueError(f"{code} has {len(draws)} control draws of size {label}, {n_controls} asked")
        for r, s in enumerate(draws):
            for alpha in ratios:
                cfgs.append({"name": f"cscale{alpha:g}.{r}_{label}@{code}", "kind": "scale", "idx": torch.tensor(s, dtype=torch.long), "langs": {index[code]}, "kw": {"alpha": float(alpha)}})
    return cfgs


def check_indices(configs, n_features=N_FEATURES):
    """Refuse a configuration whose feature set repeats an index or leaves [0, n_features)."""
    for c in configs:
        idx = c["idx"]
        if idx.numel() != torch.unique(idx).numel():
            raise ValueError(f"configuration {c['name']} repeats a feature index")
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n_features):
            raise ValueError(f"configuration {c['name']} has a feature index outside [0, {n_features})")


def register(ratios, n_controls=3, label="K64", zero=True):
    """Register the gain preset in the frozen runner and wrap its `build_configs`; returns the preset name. `zero=False` drops the zero-ablation of the group and the zeroed controls (extension C needs only
    the scaled sets). Safe to call again (the wrapper always wraps the original, never itself)."""
    ratios = validate_ratios(ratios)
    name = preset_name(ratios, n_controls, label, zero)
    fam = {"sham": [label], "scale": {label: list(ratios)}, GAIN_KEY: {"label": label, "ratios": list(ratios), "n_controls": n_controls}}
    if zero:
        fam.update({"zero": [label], "controls": [label], "n_controls": {label: n_controls}})
    fr.PRESETS[name] = fam

    def build_configs(man, run_langs, targets, panel, families, *args, **kw):
        fams = dict(families)
        gain = fams.pop(GAIN_KEY, None)
        cfgs = _FROZEN_BUILD(man, run_langs, targets, panel, fams, *args, **kw)
        if gain:
            cfgs = cfgs + control_scale_configs(man, run_langs, targets, gain["ratios"], gain["label"], gain["n_controls"])
        check_indices(cfgs, kw.get("n_features") or N_FEATURES)
        return cfgs

    fr.build_configs = build_configs
    return name


def unregister():
    """Restore the frozen `build_configs` and remove every gain preset (used by the tests)."""
    fr.build_configs = _FROZEN_BUILD
    for k in [k for k in fr.PRESETS if str(k).startswith(PRESET_PREFIX + "|")]:
        del fr.PRESETS[k]


def main(argv=None):
    """`phase1_gain_run.py --ratios R [R ...] [--n-controls 3] [--label K64] [--no-zero] <arguments of phase1_func_run.py without --preset>`: register the preset and run the frozen runner with it."""
    import argparse
    ap = argparse.ArgumentParser(description="Study G: graded-gain runs through the frozen functional-validation runner (the remaining arguments go to phase1_func_run.py).", allow_abbrev=False)
    ap.add_argument("--ratios", nargs="+", type=float, required=True); ap.add_argument("--n-controls", type=int, default=3); ap.add_argument("--label", default="K64")
    ap.add_argument("--no-zero", action="store_true", help="no zero-ablation of the group and no zeroed controls (only the scaled sets and the sham)")
    a, rest = ap.parse_known_args(argv)
    if "--preset" in rest:
        ap.error("--preset is set by this wrapper from --ratios")
    name = register(a.ratios, a.n_controls, a.label, zero=not a.no_zero)
    fr.main(rest + ["--preset", name])


if __name__ == "__main__":
    main()

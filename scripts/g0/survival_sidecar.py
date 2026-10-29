"""D-117 Stage 2: the survival sidecar. A NEW entry point (no frozen file is edited) that runs the unchanged `phase1_run.run_survival(profile_sink=...)` with a
`SurvivalSink` which streams, for every quantized condition, 21 float64/int64 statistics per (language, feature) over the equal-budget stream (D-015): the
sums behind the survival correlation, the co-active sums, the flip counts and the encoder pre-activation perturbation in threshold units. Design and gates:
reports/survival_measurement/stage2_design.md (D-119). The sink returns nothing to run_survival, so the committed pipeline's results are identical with and without it (D-035)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAT_NAMES = ("Sa", "Sb", "Saa", "Sbb", "Sab",                                              # all tokens: the sums of LangPearson
              "n_both", "Sa_both", "Sb_both", "Saa_both", "Sbb_both", "Sab_both",           # co-active tokens (a > 0 and b > 0)
              "n_a", "n_b", "n_on_off", "n_off_on",                                         # activity and flips
              "S_dpre", "S_dpre2", "S_d", "S_d2", "n_exceed2", "n_exceed1")                 # FP-active tokens, encoder side
COUNT_NAMES = ("n_both", "n_a", "n_b", "n_on_off", "n_off_on", "n_exceed2", "n_exceed1")
G2_COMMITTED = {"max_active": 1e-3, "p999_all": 1e-4, "active_threshold": 0.001, "survived": 0.9}      # pre-specified before the smoke job (stage2_design.md, update of 2026-10-29 05:30)


class SidecarError(RuntimeError):
    pass


# ------------------------------------------------------------------ token-order hash (G6)
def batch_digest(batch):
    """sha256 of one batch: language, the valid mask layout, the token ids of the valid rows and their equal-budget mask, in the order run_survival visits them."""
    valid = batch["valid"].cpu()
    h = hashlib.sha256()
    h.update(np.int64(int(batch["lang"])).tobytes()); h.update(np.asarray(valid.shape, dtype=np.int64).tobytes()); h.update(valid.numpy().astype(np.uint8).tobytes())
    h.update(batch["tokens"].cpu()[valid].numpy().astype(np.int64).tobytes()); h.update(batch["eq"].cpu()[valid].numpy().astype(np.uint8).tobytes())
    return h.hexdigest()


def digests_hash(digests):
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def batches_hash(batches):
    """The hash a condition must reproduce, computed from phase1_lib.make_batches for the same data, keep masks and batch settings."""
    return digests_hash([batch_digest(b) for b in batches])


# ------------------------------------------------------------------ the sink
class SurvivalSink:
    """Protocol of phase1_association_collect.ProfileSink: start_condition(label), update(batch, features [B, L, d]) per batch, finish_condition(). The first condition must be
    "FP": its equal-budget rows are cached sparsely (activations and pre-activations); every later condition is matched to the cache batch by batch (G6) and its
    sums are accumulated per language in float64 / int64. `pre_source()` returns the encoder pre-activation [B * L, d] of the batch that was just encoded."""
    def __init__(self, n_lang, n_feat, theta, pre_source, chunk=512):
        import phase1_lib as p1
        self.n_lang, self.n_feat, self.chunk, self.pre_source = int(n_lang), int(n_feat), int(chunk), pre_source
        self.theta = torch.as_tensor(theta, dtype=torch.float32).flatten().cpu()
        if self.theta.numel() != self.n_feat:
            raise SidecarError(f"theta has {self.theta.numel()} entries for {self.n_feat} features")
        self.sums, self.hash, self.g3_max_abs_diff = {}, {}, None
        self._cache_a, self._cache_pre, self._fp_digests = p1.SparseRows(), p1.SparseRows(), []
        self._cur, self._done, self._dev, self._theta_dev = None, [], None, None

    # -------------------------------------------------------------- protocol
    def start_condition(self, label):
        if self._cur is not None:
            raise SidecarError(f"condition {self._cur['label']} was not finished")
        if label in self._done:
            raise SidecarError(f"condition {label} was already collected")
        if label != "FP" and "FP" not in self._done:
            raise SidecarError("the FP condition must come first (its features are the reference)")
        self._cur = {"label": label, "i": 0, "digests": [], "stats": None, "n": None}

    def _init_accumulators(self, dev):
        z = lambda dt: torch.zeros(self.n_lang, self.n_feat, dtype=dt, device=dev)
        self._cur["stats"] = {k: z(torch.int64 if k in COUNT_NAMES else torch.float64) for k in STAT_NAMES}
        self._cur["n"] = torch.zeros(self.n_lang, dtype=torch.int64, device=dev)

    def update(self, batch, features):
        if self._cur is None:
            raise SidecarError("update outside a condition: call start_condition first")
        valid_shape = tuple(batch["valid"].shape)
        if features.ndim != 3 or features.shape[-1] != self.n_feat or tuple(features.shape[:2]) != valid_shape:
            raise SidecarError(f"features must have shape {valid_shape} + ({self.n_feat},), got {tuple(features.shape)}")
        pre = self.pre_source()
        flat = features.reshape(-1, self.n_feat)
        if pre is None or tuple(pre.shape) != tuple(flat.shape):
            raise SidecarError(f"no pre-activation of shape {tuple(flat.shape)} for this batch (got {None if pre is None else tuple(pre.shape)})")
        dev = features.device
        if self._dev is None or self._dev != dev:
            self._dev, self._theta_dev = dev, self.theta.to(dev)
        valid = batch["valid"].reshape(-1).to(dev); sel = batch["eq"].reshape(-1).to(dev)[valid]
        a_v, pre_v = flat[valid], pre.to(dev)[valid]
        if self._cur["i"] == 0:
            diff = gate_g3(a_v, pre_v, self._theta_dev)                                         # G3: the SAE arithmetic, on the first batch of every condition
            self.g3_max_abs_diff = diff if self.g3_max_abs_diff is None else max(self.g3_max_abs_diff, diff)
        digest = batch_digest(batch); i = self._cur["i"]; self._cur["digests"].append(digest)
        x, px = a_v[sel], pre_v[sel]
        if self._cur["label"] == "FP":
            self._fp_digests.append(digest)
            self._cache_a.add(x); self._cache_pre.add(torch.where(x > 0, px, torch.zeros_like(px)))
        else:
            if i >= len(self._fp_digests) or digest != self._fp_digests[i]:
                raise SidecarError(f"{self._cur['label']}: batch {i} is not the batch {i} of the FP pass (order, language, tokens or equal-budget mask differ); the FP cache would be misaligned")
            a, pf = self._cache_a.get(i, dev), self._cache_pre.get(i, dev)
            if tuple(a.shape) != tuple(x.shape):
                raise SidecarError(f"{self._cur['label']}: batch {i} holds {x.shape[0]} equal-budget rows, the FP pass cached {a.shape[0]}")
            if self._cur["stats"] is None:
                self._init_accumulators(dev)
            lang = int(batch["lang"])
            for s in range(0, a.shape[0], self.chunk):
                self._accumulate(lang, a[s:s + self.chunk], x[s:s + self.chunk], pf[s:s + self.chunk], px[s:s + self.chunk])
            self._cur["n"][lang] += a.shape[0]
        self._cur["i"] += 1

    def _accumulate(self, lang, a, b, pf, pq):
        st, th = self._cur["stats"], self._theta_dev
        on_a, on_b = a > 0, b > 0
        a64, b64 = a.double(), b.double()
        st["Sa"][lang] += a64.sum(0); st["Sb"][lang] += b64.sum(0); st["Saa"][lang] += (a64 ** 2).sum(0); st["Sbb"][lang] += (b64 ** 2).sum(0); st["Sab"][lang] += (a64 * b64).sum(0)
        both = on_a & on_b; m = both.double(); am, bm = a64 * m, b64 * m
        st["n_both"][lang] += both.sum(0); st["Sa_both"][lang] += am.sum(0); st["Sb_both"][lang] += bm.sum(0)
        st["Saa_both"][lang] += (am * a64).sum(0); st["Sbb_both"][lang] += (bm * b64).sum(0); st["Sab_both"][lang] += (am * b64).sum(0)
        st["n_a"][lang] += on_a.sum(0); st["n_b"][lang] += on_b.sum(0); st["n_on_off"][lang] += (on_a & ~on_b).sum(0); st["n_off_on"][lang] += (~on_a & on_b).sum(0)
        act = on_a.double()                                                                      # FP-active tokens only; the differences of float32 numbers are exact in float64
        dpre = (pq.double() - pf.double()) * act; dd = (pf.double() - th.double()) * act
        st["S_dpre"][lang] += dpre.sum(0); st["S_dpre2"][lang] += (dpre ** 2).sum(0); st["S_d"][lang] += dd.sum(0); st["S_d2"][lang] += (dd ** 2).sum(0)
        st["n_exceed2"][lang] += (on_a & (dpre.abs() > dd)).sum(0)
        st["n_exceed1"][lang] += (on_a & ~(pq > th)).sum(0)                                      # the SAE's own comparison: a tie at theta is off

    def finish_condition(self):
        if self._cur is None:
            raise SidecarError("finish_condition outside a condition")
        label, digests = self._cur["label"], self._cur["digests"]
        if label != "FP" and len(digests) != len(self._fp_digests):
            raise SidecarError(f"{label}: {len(digests)} batches, the FP pass had {len(self._fp_digests)}")
        self.hash[label] = digests_hash(digests)
        if label != "FP":
            if self._cur["stats"] is None:
                self._init_accumulators(self._dev if self._dev is not None else "cpu")
            self.sums[label] = {"label": label, "stats": {k: v.cpu() for k, v in self._cur["stats"].items()}, "n": self._cur["n"].cpu()}
        self._done.append(label); self._cur = None


# ------------------------------------------------------------------ derived quantities and gates
def corr_from_sums(sums):
    """The LangPearson formula (float64, denominator clamped at sqrt(1e-12), result clamped to [-1, 1]) applied to the saved sums -> [n_lang, d]."""
    st, n = sums["stats"], sums["n"].double()[:, None]
    out = torch.zeros(st["Sa"].shape, dtype=torch.float64)
    ok = (n > 0).expand_as(out)
    num = st["Sab"] - st["Sa"] * st["Sb"] / n.clamp(min=1)
    den = torch.sqrt(torch.clamp((st["Saa"] - st["Sa"] ** 2 / n.clamp(min=1)) * (st["Sbb"] - st["Sb"] ** 2 / n.clamp(min=1)), min=1e-12))
    return torch.where(ok, torch.clamp(num / den, -1.0, 1.0), out)


def gate_g3(acts, pre, theta, tol=1e-5):
    """G3: acts == pre * (pre > theta); returns the max absolute difference, raises above tol."""
    expected = pre * (pre > theta.to(pre.device))
    diff = float((acts.to(expected.dtype) - expected).abs().max()) if acts.numel() else 0.0
    if not diff <= tol:
        raise SidecarError(f"G3 failed: acts differ from pre * (pre > theta) by {diff} (tolerance {tol}): wrong pre-activation source or threshold")
    return diff


def gate_int0(sums, rtol=1e-12):
    """G1: the INT0 condition (restore only) streams the identity: no flips, no perturbation, and every b-sum equals the a-sum."""
    st, bad = sums["stats"], []
    for k in ("n_on_off", "n_off_on", "n_exceed1", "n_exceed2"):
        if int(st[k].abs().sum()):
            bad.append(f"{k} is not 0")
    for k in ("S_dpre", "S_dpre2"):
        if float(st[k].abs().sum()) != 0.0:
            bad.append(f"{k} is not 0")
    if not (torch.equal(st["n_both"], st["n_a"]) and torch.equal(st["n_a"], st["n_b"])):
        bad.append("n_both, n_a, n_b differ")
    pairs = [("Sa", "Sb"), ("Saa", "Sbb"), ("Saa", "Sab"), ("Sa", "Sa_both"), ("Sb", "Sb_both"), ("Saa", "Saa_both"), ("Sbb", "Sbb_both"), ("Sab", "Sab_both")]
    for x, y in pairs:
        if not torch.allclose(st[x], st[y], rtol=rtol, atol=0.0):
            bad.append(f"{x} != {y} beyond relative {rtol}")
    if bad:
        raise SidecarError("G1 failed (INT0 is not the identity): " + "; ".join(bad))


def gate_g2_internal(sums, corr_run, tol=1e-7):
    """G2(i): r rebuilt from the sums (float64) equals the correlation run_survival returned in the same job; returns the maximum absolute difference. run_survival's
    `_pack` stores the correlation as float32, so the comparison cannot be tighter than the float32 rounding (half a unit in the last place is 6e-8 near 1): 1e-7."""
    diff = float((corr_from_sums(sums) - corr_run.double()).abs().max())
    if not diff <= tol:
        raise SidecarError(f"G2(i) failed: r from the sums differs from run_survival's r by {diff} (tolerance {tol})")
    return diff


def gate_g2_committed(corr, committed_corr, committed_firing_rate):
    """G2(ii), pre-specified: per language the number of active features (FP firing rate above 0.001) with r above 0.9 is identical to the committed one; the largest
    |difference| over active features is at most 1e-3 and the 99.9th percentile over ALL cells is at most 1e-4. Never raises: the caller decides (stop / investigate)."""
    c, k = corr.double(), committed_corr.double()
    active = committed_firing_rate > G2_COMMITTED["active_threshold"]; diff = (c - k).abs()
    mine, theirs = ((c > G2_COMMITTED["survived"]) & active).sum(1), ((k > G2_COMMITTED["survived"]) & active).sum(1)
    max_active = float(diff[active].max()) if bool(active.any()) else 0.0
    p999 = float(torch.quantile(diff.flatten(), 0.999)) if diff.numel() else 0.0
    identical = bool(torch.equal(mine, theirs))
    return {"passed": identical and max_active <= G2_COMMITTED["max_active"] and p999 <= G2_COMMITTED["p999_all"], "counts_identical": identical, "max_abs_diff_active": max_active,
            "p999_abs_diff_all": p999, "max_abs_diff_all": float(diff.max()) if diff.numel() else 0.0, "per_language_survived": mine.tolist(), "per_language_survived_committed": theirs.tolist()}


def gate_g4(sums, committed_firing_rate=None):
    """G4: n_on_off equals n_exceed1 exactly in every (language, feature) (a flip off is exactly `not (pre_Q > theta)` on an FP-active token); with the committed FP firing
    rate, |n_a - rate * n| <= 1 (the rate is stored as a float)."""
    st, n = sums["stats"], sums["n"]
    if not torch.equal(st["n_on_off"], st["n_exceed1"]):
        raise SidecarError(f"G4 failed: n_on_off differs from n_exceed1 in {int((st['n_on_off'] != st['n_exceed1']).sum())} cells")
    if committed_firing_rate is not None:
        gap = (st["n_a"].double() - committed_firing_rate.double() * n.double()[:, None]).abs()
        if not float(gap.max()) <= 1.0 + 1e-9:
            raise SidecarError(f"G4 failed: n_a differs from the committed firing rate times n by up to {float(gap.max())} tokens")


# ------------------------------------------------------------------ files
def save_sums(out_dir, label, sink):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = sink.sums[label]
    torch.save({"label": label, "stat_names": list(STAT_NAMES), "stats": s["stats"], "n": s["n"]}, out / f"sums_{label}.pt")


def load_sums(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_dir(out_dir, conditions, n_lang, n_feat):
    """Completeness and consistency of a job's saved sums; raises SidecarError on the first problem. Returns {"conditions", "n"}."""
    out, n_ref = Path(out_dir), None
    for label in conditions:
        f = out / f"sums_{label}.pt"
        if not f.exists():
            raise SidecarError(f"missing file {f.name}")
        d = load_sums(f)
        if set(d) != {"label", "stat_names", "stats", "n"} or d["label"] != label or list(d["stat_names"]) != list(STAT_NAMES):
            raise SidecarError(f"{f.name}: unexpected structure")
        if set(d["stats"]) != set(STAT_NAMES):
            raise SidecarError(f"{f.name}: statistics {sorted(set(STAT_NAMES) ^ set(d['stats']))} missing or extra")
        for k, v in d["stats"].items():
            want = torch.int64 if k in COUNT_NAMES else torch.float64
            if v.dtype != want or tuple(v.shape) != (n_lang, n_feat):
                raise SidecarError(f"{f.name}: {k} is {v.dtype} {tuple(v.shape)}, expected {want} {(n_lang, n_feat)}")
            if not bool(torch.isfinite(v.double()).all()) or (k in COUNT_NAMES and int(v.min()) < 0):
                raise SidecarError(f"{f.name}: {k} holds non-finite or negative counts")
        n = d["n"]
        if n.dtype != torch.int64 or tuple(n.shape) != (n_lang,):
            raise SidecarError(f"{f.name}: n is {n.dtype} {tuple(n.shape)}")
        if int(d["stats"]["n_a"].max()) > int(n.max()) or int(d["stats"]["n_b"].max()) > int(n.max()):
            raise SidecarError(f"{f.name}: an activity count exceeds the token count")
        if n_ref is None:
            n_ref = n
        elif not torch.equal(n, n_ref):
            raise SidecarError(f"{f.name}: token counts differ from the first condition's (the conditions must see the same tokens)")
    return {"conditions": list(conditions), "n": n_ref.tolist() if n_ref is not None else []}


# ------------------------------------------------------------------ pre-activation capture
class PreCapture:
    """The encoder pre-activation of the batch just encoded. Source: the SAE's own hook point `hook_sae_acts_pre` through a forward hook (sae_lens 6.x); fallback for an SAE
    without it: x @ W_enc + b_enc with the SAE's input processing (b_dec subtracted when cfg.apply_b_dec_to_input). Gate G3 checks acts == pre * (pre > theta) either way."""
    def __init__(self, sae):
        self.sae, self.last, self.dtype, self._handle = sae, None, None, None
        hook = getattr(sae, "hook_sae_acts_pre", None)
        if hook is not None and hasattr(hook, "register_forward_hook"):
            self.source = "sae.hook_sae_acts_pre"
            self._handle = hook.register_forward_hook(lambda module, inputs, output: self._set(output))
        else:
            self.source = "x@W_enc+b_enc"

    def _set(self, pre):
        self.last, self.dtype = pre.detach(), str(pre.dtype)

    def _fallback(self, acts):
        s = self.sae
        x = acts.to(s.W_enc.device).to(s.W_enc.dtype)
        if getattr(getattr(s, "cfg", None), "apply_b_dec_to_input", False):
            x = x - s.b_dec
        return x @ s.W_enc + s.b_enc

    def wrap(self, encode):
        def wrapped(acts):
            self.last = None
            out = encode(acts)
            if self.source != "sae.hook_sae_acts_pre":
                self._set(self._fallback(acts))
            return out
        return wrapped

    def get(self):
        if self.last is None:
            raise SidecarError("no pre-activation was captured for the last batch (the SAE's encode did not pass through the hook point)")
        return self.last

    def close(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None
        self.last = None


# ------------------------------------------------------------------ the gates of a finished run, thresholds, committed reference
def check_thresholds(theta, csv_path, rtol=1e-6, atol=1e-6):
    """G3 (second half): the thresholds of the loaded SAE equal the committed table reports/phase1/sae_thresholds.csv; returns the maximum absolute difference."""
    rows = [ln.strip().split(",") for ln in Path(csv_path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    if rows[0] != ["feature", "threshold"]:
        raise SidecarError(f"{csv_path}: unexpected header {rows[0]}")
    table = torch.tensor([float(r[1]) for r in rows[1:]], dtype=torch.float64)
    if [int(r[0]) for r in rows[1:]] != list(range(len(table))) or table.numel() != theta.numel():
        raise SidecarError(f"{csv_path}: {table.numel()} thresholds for {theta.numel()} features, or the feature index is not 0..n-1")
    t = theta.double().cpu()
    if not torch.allclose(t, table, rtol=rtol, atol=atol):
        raise SidecarError(f"G3 failed: the SAE thresholds differ from {Path(csv_path).name} by up to {float((t - table).abs().max())}")
    return float((t - table).abs().max())


def load_committed(run_dir):
    """The committed survival.pt of an arm's stage-1 run (`.../mask/eq`): corr {bits: [n_lang, d] float32}, firing_rate [n_lang, d], languages, n."""
    path = Path(run_dir) / "survival.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                                                                    # our own committed artifact; only older pickles need the full loader
        return torch.load(path, map_location="cpu", weights_only=False)


def check_committed(job_dir, committed_run):
    """G2(ii) and the committed half of G4 on a FINISHED job's saved sums (run locally after the files are fetched, so the committed survival.pt need not be on the cluster):
    -> {"languages_match", "G2_committed": {"passed", "per_condition"}, "G4_committed": {condition: bool}}. Never raises on a numeric disagreement; the caller reports it."""
    job = Path(job_dir); meta = json.loads((job / "meta.json").read_text()); committed = load_committed(committed_run)
    out = {"languages_match": list(committed["languages"]) == list(meta["languages"]), "G2_committed": {"per_condition": {}}, "G4_committed": {}}
    for label in meta["saved_conditions"]:
        sums, bits = load_sums(job / f"sums_{label}.pt"), int(label[3:])
        if bits not in committed["corr"]:
            continue
        out["G2_committed"]["per_condition"][label] = gate_g2_committed(corr_from_sums(sums), committed["corr"][bits], committed["firing_rate"])
        try:
            gate_g4(sums, committed_firing_rate=committed["firing_rate"]); out["G4_committed"][label] = True
        except SidecarError:
            out["G4_committed"][label] = False
    out["G2_committed"]["passed"] = bool(out["languages_match"]) and all(v["passed"] for v in out["G2_committed"]["per_condition"].values())
    return out


def run_gates(sink, res_eq, bits, expected_hash, theta_csv=None, committed=None):
    """Every gate of the design on a finished run. -> (gates, failures, warnings): `failures` are fatal (the job ends non-zero after writing its files), `warnings` hold G2(ii)
    (the committed comparison: investigated, never tolerated silently, but the sidecar's own output stays analysable)."""
    labels = [f"INT{b}" for b in bits]; gates, failures, warnings = {}, [], []

    def attempt(name, fn):
        try:
            gates[name] = {"passed": True, **fn()}
        except SidecarError as e:
            gates[name] = {"passed": False, "why": str(e)}; failures.append(name)
    def g6():
        if sink.hash.get("FP") != expected_hash:
            raise SidecarError("the FP pass hashes differently from make_batches for the same data")
        if any(sink.hash.get(c) != sink.hash["FP"] for c in labels):
            raise SidecarError("a condition saw other tokens than the FP pass")
        return {"hash": expected_hash}
    attempt("G6", g6)

    def g3():
        out = {"max_abs_diff_acts_vs_pre": sink.g3_max_abs_diff}
        if theta_csv is not None:
            out["max_abs_diff_theta_vs_csv"] = check_thresholds(sink.theta, theta_csv)
        return out
    attempt("G3", g3)

    def g1():
        if "INT0" not in sink.sums:
            return {"note": "no INT0 condition in this job"}
        gate_int0(sink.sums["INT0"])
        return {}
    attempt("G1", g1)
    attempt("G2_internal", lambda: {"max_abs_diff": {c: gate_g2_internal(sink.sums[c], res_eq[int(c[3:])]["corr"]) for c in labels if c in sink.sums}})

    def g4():
        for c in labels:
            if c in sink.sums:
                gate_g4(sink.sums[c], committed_firing_rate=res_eq["FP"]["firing_rate"])
        if committed is not None:
            for c in labels:
                if c in sink.sums:
                    gate_g4(sink.sums[c], committed_firing_rate=committed["firing_rate"])
        return {"checked_against_committed": committed is not None}
    attempt("G4", g4)
    if committed is not None:
        per = {c: gate_g2_committed(corr_from_sums(sink.sums[c]), committed["corr"][int(c[3:])], committed["firing_rate"]) for c in labels if c in sink.sums and int(c[3:]) in committed["corr"]}
        gates["G2_committed"] = {"passed": all(v["passed"] for v in per.values()), "per_condition": per}
        if not gates["G2_committed"]["passed"]:
            warnings.append("G2_committed")
    return gates, failures, warnings


# ------------------------------------------------------------------ the entry point (run through gptq_ext/run_with_weights.py --entry survival_sidecar:main)
ROOT = Path(__file__).resolve().parents[2]
SAE_RELEASE, SAE_ID, LAYER = "gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical", 12                  # = phase1_run's constants
MODEL_CANDIDATES = ["gemma-2-2b", "google/gemma-2-2b"]


def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(description="D-117 Stage 2: survival sidecar (21 streamed statistics per condition, language and feature; equal-budget stream; mask regime).")
    ap.add_argument("--out", required=True); ap.add_argument("--bits", type=int, nargs="+", required=True, help="rungs in the order to run, e.g. 16 6 4 0; INT0 is a gate only and is not saved")
    ap.add_argument("--flores-root", required=True); ap.add_argument("--roster", default=str(ROOT / "reports/g0/phase1_languages.json"))
    ap.add_argument("--languages", nargs="*", help="subset of FLORES codes (smoke)"); ap.add_argument("--max-sentences", type=int, help="first N devtest sentences per language (smoke)")
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--commit", default="unknown")
    ap.add_argument("--committed-run", help="the arm's committed mask/eq run directory (survival.pt) for gate G2(ii); ignored for subset (smoke) runs")
    ap.add_argument("--thresholds-csv", default=str(ROOT / "reports/phase1/sae_thresholds.csv"))
    a = ap.parse_args(argv)
    if len(set(a.bits)) != len(a.bits) or any(b < 0 for b in a.bits):
        ap.error("--bits must be distinct non-negative integers")
    return a


def main(argv=None):
    import gc, time
    a = parse_args(argv)
    t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:7.0f}s]", *m, flush=True)
    import numpy as np
    from tqdm.auto import tqdm
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    import phase1_run as run
    import phase1_data as pd_
    import their_code as tc
    torch.set_grad_enabled(False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = {"torch": torch, "np": np, "tqdm": tqdm, "gc": gc, "HookedTransformer": HookedTransformer, "SAE": SAE, "DEVICE": DEVICE, "WORK_DEVICE": DEVICE, "BATCH_SIZE": 1, "FIRING_THRESHOLD": run.FIRING_THRESHOLD}
    ex = tc.extract_functions(tc.PRISTINE, tc.THEIR_FUNCTIONS); tc.verify_hashes(ex, json.loads(tc.MANIFEST.read_text())["functions"]); tc.build_namespace(ex, env)
    log("their functions verified against the frozen manifest:", sorted(ex))
    model, model_name = env["load_gemma_tl_model"](MODEL_CANDIDATES)
    sae = env["load_sae_compat"](SAE_RELEASE, SAE_ID, DEVICE)
    hook = getattr(sae.cfg, "hook_name", None) or f"blocks.{LAYER}.hook_resid_post"
    assert sae.cfg.d_in == model.cfg.d_model, "SAE / model width mismatch"
    n_feat = int(sae.cfg.d_sae)
    env["model_work"] = model; env["original_state_cpu"] = env["clone_state_to_cpu"](model)
    tok = model.tokenizer; bos = tok.bos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    log("model", model_name, "hook", hook, "d_sae", n_feat, "bos", bos, "pad", pad)

    def forward(tokens):
        logits, cache = model.run_with_cache(tokens, names_filter=[hook]); return cache[hook], logits
    quantize = lambda bits: env["quantize_rtn_per_output_channel_tl"](model, bits=bits)
    restore = env["restore_model_work"]

    roster = pd_.load_roster(a.roster)
    if a.languages:
        roster = [r for r in roster if r["code"] in set(a.languages)]
    devtest, rep_test = pd_.build_corpus(a.flores_root, roster, "devtest", tok, None if a.languages else 1012)
    if a.max_sentences:
        devtest = {k: v[:a.max_sentences] for k, v in devtest.items()}
    names = [r["code"] for r in roster]
    if not hasattr(sae, "threshold"):
        raise SidecarError("the loaded SAE has no `threshold` parameter (not a JumpReLU SAE)")
    theta = sae.threshold.detach().float().cpu()
    cap = PreCapture(sae); encode = cap.wrap(lambda acts: env["encode_sae"](sae, acts))
    meta_base = {"model": model_name, "sae_release": SAE_RELEASE, "sae_id": SAE_ID, "hook": hook, "device": DEVICE,
                 "versions": run.package_versions(["torch", "transformer-lens", "sae-lens", "transformers", "datasets"])}
    run_job(a, names, devtest, bos, pad, n_feat, forward, encode, quantize, restore, theta, cap, meta_base, log, t0, argv)


def run_job(a, names, devtest, bos, pad, n_feat, forward, encode, quantize, restore, theta, cap, meta_base, log, t0, argv=None):
    """Everything after the model and the SAE are loaded: the equal-budget masks, run_survival with the sink, the gates, the files and meta.json. The toy harness drives exactly this
    function in the tests (a stand-in for the model, the SAE and `cap`)."""
    import time
    import phase1_lib as p1
    import phase1_run as run
    device = meta_base.get("device", "cpu")
    budget = p1.budget_for(devtest); keep = p1.keep_masks(devtest, budget, a.seed, split_offset=2)                              # exactly phase1_run's devtest equal-budget masks (D-015)
    expected_hash = batches_hash(p1.make_batches(devtest, bos, "mask", pad, 256, a.batch_size, a.max_tokens, keep))
    log("languages:", len(names), "devtest sentences:", sum(len(v) for v in devtest.values()), "equal budget per language:", budget, "batch hash", expected_hash[:16])
    sink = SurvivalSink(len(names), n_feat, theta, cap.get)
    theta_sha = hashlib.sha256(theta.float().cpu().numpy().tobytes()).hexdigest()
    log("pre-activation source:", cap.source, "; theta sha256", theta_sha[:16])
    results = run.run_survival(forward, encode, quantize, restore, devtest, "mask", bos, pad, n_feat, a.bits, device=device, batch_size=a.batch_size, max_tokens=a.max_tokens, keep=keep, profile_sink=sink)
    cap.close()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    log("survival pass with the sidecar done; peak GPU memory GB:", peak)
    _, res_eq = run.split_streams(results)

    committed = load_committed(a.committed_run) if (a.committed_run and not a.languages) else None
    if a.committed_run and a.languages:
        log("subset run: the committed comparison (G2(ii)) is skipped")
    gates, failures, warnings = run_gates(sink, res_eq, a.bits, expected_hash, a.thresholds_csv, committed)
    log("gates:", {k: v["passed"] for k, v in gates.items()}, "failures", failures, "warnings", warnings)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    saved = [f"INT{b}" for b in a.bits if b != 0]
    for label in saved:
        save_sums(out, label, sink)
    meta = {"study": "D-117 survival sidecar (Stage 2)", "design": "reports/survival_measurement/stage2_design.md", "commit": a.commit, "argv": list(argv) if argv is not None else None,
            **meta_base, "regime": "mask", "stream": "eq", "bits": a.bits, "saved_conditions": saved, "languages": names, "n_feat": n_feat,
            "tokens_per_language": sink.sums[saved[0]]["n"].tolist() if saved else [], "equal_budget_per_language": budget, "batch_size": a.batch_size, "max_tokens": a.max_tokens, "seed": a.seed,
            "subset": bool(a.languages), "max_sentences": a.max_sentences, "peak_gpu_gb": peak, "seconds": time.time() - t0, "pre_activation_source": cap.source, "pre_activation_dtype": cap.dtype,
            "theta_sha256": theta_sha, "theta": theta.tolist(), "batch_hash": expected_hash, "stat_names": list(STAT_NAMES), "gates": gates, "gate_failures": failures, "gate_warnings": warnings,
            "committed_run": a.committed_run if committed is not None else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    validate_dir(out, saved, n_lang=len(names), n_feat=n_feat)
    log("saved", out, "conditions", saved)
    if failures:
        raise SystemExit(f"gates failed: {failures} (the files are written for diagnosis; nothing is claimed from this job)")
    if warnings:
        log("WARNING: the committed comparison G2(ii) failed:", warnings, "- investigate before any analysis; the sidecar's own sums are internally consistent (G2(i) passed)")
    return meta


if __name__ == "__main__":
    main(sys.argv[1:])

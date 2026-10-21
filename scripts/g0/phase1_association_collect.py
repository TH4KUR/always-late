"""D-035 Task 2 (plan: docs/superpowers/plans/2026-10-21-language-association-retention.md): the optional per-condition response collector of the association-retention study.
`ProfileSink` is passed to phase1_run.run_survival(profile_sink=...) and receives, for every condition (FP, INTb, INT0), the encoder outputs of THAT condition for every batch.
It saves, per stream (`eq` = equal-budget positions, `all` = every content position; BOS and padding never count) and condition:
  association/<stream>/profiles.pt  {"schema_version": 1, "languages": [codes], "feature_ids": int64 [F], "conditions": {cond: {"n_tokens": int64 [L],
      "positive_counts": int64 [L, F] (z > 0), "activation_sums": float64 [L, F], "activation_max": float64 [L, F]}}}
  association/<stream>/sentences/<cond>/<language>.pt   per-sentence totals of a fixed feature subset as CSR rows: sentence_ids (index in the language, ascending), n_tokens
      (selected content tokens of the sentence, 0 allowed), feature_ids (the subset), row_ptr int64 [S + 1], col int32 (index into feature_ids), counts int16, sums float64
  association/manifest.json   completion state, languages, conditions, shard index with sha256 / size / nnz, bytes_total, configuration.
The quantities are computed from the condition's own activations, never from the reference-side statistics of phase1_lib.LangPearson. Positive means strictly greater than zero.
A run directory must be fresh (no resume protocol). `validate_run` fails on an incomplete run, a wrong roster, a wrong sentence count or a modified file."""
import hashlib, json
from pathlib import Path

import torch

SCHEMA_VERSION = 1
FIELDS = ("n_tokens", "positive_counts", "activation_sums", "activation_max")


class CollectError(ValueError):
    pass


def condition_name(x):
    return "FP" if x == "FP" else f"INT{int(x)}"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ProfileSink:
    def __init__(self, out_dir, languages, n_feat, conditions, sentence_features, streams=("eq", "all"), config=None, sentence_streams=None, sums_dtype="float64"):
        self.root = Path(out_dir) / "association"
        if self.root.exists() and any(self.root.iterdir()):
            raise CollectError(f"{self.root} is not empty: a collection needs a fresh output directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.languages, self.n_feat, self.conditions, self.streams = list(languages), int(n_feat), list(conditions), tuple(streams)
        sf = torch.as_tensor(sentence_features, dtype=torch.long).flatten()
        if sf.numel() and (int(sf.min()) < 0 or int(sf.max()) >= self.n_feat or len(torch.unique(sf)) != sf.numel()):
            raise CollectError("sentence_features must be distinct feature ids below n_feat")
        self.sentence_features = torch.sort(sf).values
        self.sentence_streams = tuple(self.streams if sentence_streams is None else sentence_streams)
        if not set(self.sentence_streams) <= set(self.streams):
            raise CollectError(f"sentence_streams {self.sentence_streams} must be among the streams {self.streams}")
        if sums_dtype not in ("float64", "float32"):
            raise CollectError("sums_dtype must be float64 or float32")
        self.sums_dtype = getattr(torch, sums_dtype)                                  # per-sentence sums only; the aggregate profiles are always float64
        self.config = {**dict(config or {}), "sentence_streams": list(self.sentence_streams), "sums_dtype": sums_dtype}
        self._cur, self._done, self._profiles, self._shards = None, [], {s: {} for s in self.streams}, {s: {} for s in self.sentence_streams}

    # ---------------------------------------------------------------- collection protocol
    def start_condition(self, condition):
        if condition not in self.conditions:
            raise CollectError(f"unknown condition {condition!r}; expected one of {self.conditions}")
        if self._cur is not None:
            raise CollectError(f"condition {self._cur['name']} was not finished")
        if condition in self._done:
            raise CollectError(f"condition {condition} was already collected")
        n_l = len(self.languages)
        self._cur = {"name": condition, "dev": None, "lang": None, "finished_langs": set(),
                     "agg": {s: None for s in self.streams}, "pieces": {s: [] for s in self.streams}}
        self._n_l = n_l

    def _init_agg(self, dev):
        L, F = self._n_l, self.n_feat
        self._cur["dev"] = dev; self._sf = self.sentence_features.to(dev)
        for s in self.streams:
            self._cur["agg"][s] = {"n": torch.zeros(L, dtype=torch.long, device=dev), "cnt": torch.zeros(L, F, dtype=torch.long, device=dev),
                                   "sum": torch.zeros(L, F, dtype=torch.float64, device=dev), "max": torch.zeros(L, F, dtype=torch.float64, device=dev)}

    def update(self, batch, features):
        """batch: a phase1_lib.make_batches batch (lang, valid, eq, sent); features: encoder outputs of this condition, shape [batch, position, n_feat]."""
        if self._cur is None:
            raise CollectError("update outside a condition: call start_condition first")
        if features.ndim != 3 or features.shape[-1] != self.n_feat:
            raise CollectError(f"features must have shape [batch, position, {self.n_feat}], got {tuple(features.shape)}")
        dev = features.device
        if self._cur["dev"] is None:
            self._init_agg(dev)
        lang = int(batch["lang"])
        if self._cur["lang"] != lang:
            if lang in self._cur["finished_langs"]:
                raise CollectError(f"batches of language {lang} must be contiguous")
            if self._cur["lang"] is not None:
                self._flush(self._cur["lang"])
            self._cur["lang"] = lang
        valid = batch["valid"].to(dev)
        if valid.shape != features.shape[:2]:
            raise CollectError("valid mask and features disagree in shape")
        ids = torch.as_tensor(batch["sent"]).to(dev).long(); B = valid.shape[0]
        for s in self.streams:
            mask = valid if s == "all" else (batch["eq"].to(dev) & valid)
            rows = features[mask]; a = self._cur["agg"][s]
            a["n"][lang] += rows.shape[0]
            if rows.shape[0]:
                a["cnt"][lang] += (rows > 0).sum(0); a["sum"][lang] += rows.double().sum(0); a["max"][lang] = torch.maximum(a["max"][lang], rows.max(0).values.double())
            if s not in self.sentence_streams:
                continue
            idx = torch.arange(B, device=dev)[:, None].expand_as(mask)[mask]
            sub = rows[:, self._sf]
            cnt_b = torch.zeros(B, sub.shape[1], dtype=torch.int32, device=dev).index_add_(0, idx, (sub > 0).int())       # int32: half the transient GPU memory of int64
            sum_b = torch.zeros(B, sub.shape[1], dtype=torch.float64, device=dev).index_add_(0, idx, sub.double())
            if cnt_b.numel() and int(cnt_b.max()) > 32767:
                raise CollectError("a per-sentence count exceeds the int16 range")
            r, c = ((cnt_b > 0) | (sum_b != 0)).nonzero(as_tuple=True)
            lens = torch.bincount(r, minlength=B)
            self._cur["pieces"][s].append({"ids": ids.cpu(), "ntok": mask.sum(1).long().cpu(), "lens": lens.cpu(), "col": c.int().cpu(), "cnt": cnt_b[r, c].short().cpu(), "sum": sum_b[r, c].to(self.sums_dtype).cpu()})

    def _flush(self, lang):
        for s in self.sentence_streams:
            pcs = self._cur["pieces"][s]
            if not pcs:
                continue
            ids = torch.cat([p["ids"] for p in pcs]); ntok = torch.cat([p["ntok"] for p in pcs]); lens = torch.cat([p["lens"] for p in pcs])
            col = torch.cat([p["col"] for p in pcs]); cnt = torch.cat([p["cnt"] for p in pcs]); sm = torch.cat([p["sum"] for p in pcs])
            if len(torch.unique(ids)) != len(ids):
                raise CollectError(f"duplicate sentence ids in language {self.languages[lang]}")
            start = torch.cumsum(lens, 0) - lens; order = torch.argsort(ids)
            new_lens = lens[order]; new_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(new_lens, 0)])
            total = int(new_ptr[-1])
            gather = torch.arange(total) - torch.repeat_interleave(new_ptr[:-1], new_lens) + torch.repeat_interleave(start[order], new_lens)
            shard = {"schema_version": SCHEMA_VERSION, "condition": self._cur["name"], "stream": s, "language": self.languages[lang], "sentence_ids": ids[order],
                     "n_tokens": ntok[order], "feature_ids": self.sentence_features.cpu(), "row_ptr": new_ptr, "col": col[gather], "counts": cnt[gather], "sums": sm[gather]}
            path = self.root / s / "sentences" / self._cur["name"] / f"{self.languages[lang]}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(shard, path)
            self._shards[s].setdefault(self._cur["name"], {})[self.languages[lang]] = {"file": str(path.relative_to(self.root)), "sha256": _sha256(path), "bytes": path.stat().st_size,
                                                                                         "n_sentences": int(len(ids)), "nnz": total}
            self._cur["pieces"][s] = []
        self._cur["finished_langs"].add(lang)

    def finish_condition(self):
        if self._cur is None:
            raise CollectError("finish_condition without start_condition")
        if self._cur["lang"] is not None:
            self._flush(self._cur["lang"])
        for s in self.streams:
            a = self._cur["agg"][s]
            if a is not None:
                self._profiles[s][self._cur["name"]] = {"n_tokens": a["n"].cpu(), "positive_counts": a["cnt"].cpu(), "activation_sums": a["sum"].cpu(), "activation_max": a["max"].cpu()}
        self._done.append(self._cur["name"]); self._cur = None

    def finalize(self):
        if self._cur is not None:
            raise CollectError(f"condition {self._cur['name']} was not finished")
        files = {}
        for s in self.streams:
            conds = {c: self._profiles[s][c] for c in self.conditions if c in self._profiles[s]}
            path = self.root / s / "profiles.pt"; path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": SCHEMA_VERSION, "languages": list(self.languages), "feature_ids": torch.arange(self.n_feat, dtype=torch.long), "conditions": conds}, path)
            files[s] = {"file": str(path.relative_to(self.root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        missing_conditions = [c for c in self.conditions if c not in self._done]
        missing_shards = [f"{s}/{c}/{l}" for s in self.sentence_streams for c in self.conditions for l in self.languages if l not in self._shards[s].get(c, {})]
        bytes_total = sum(f["bytes"] for f in files.values()) + sum(x["bytes"] for s in self._shards.values() for c in s.values() for x in c.values())
        manifest = {"schema_version": SCHEMA_VERSION, "complete": not missing_conditions and not missing_shards, "missing_conditions": missing_conditions, "missing_shards": missing_shards,
                    "languages": list(self.languages), "n_features": self.n_feat, "conditions": list(self.conditions), "streams": list(self.streams),
                    "sentence_features": int(self.sentence_features.numel()), "sentence_features_sha256": hashlib.sha256(self.sentence_features.numpy().tobytes()).hexdigest(),
                    "profiles": files, "shards": self._shards, "bytes_total": bytes_total, "config": self.config}
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return manifest


# ---------------------------------------------------------------- reading and validation
def read_profiles(out_dir, stream):
    return torch.load(Path(out_dir) / "association" / stream / "profiles.pt", map_location="cpu", weights_only=True)


def read_shard(out_dir, stream, condition, language):
    return torch.load(Path(out_dir) / "association" / stream / "sentences" / condition / f"{language}.pt", map_location="cpu", weights_only=True)


def _dense(shard, values, dtype):
    S, K = len(shard["sentence_ids"]), len(shard["feature_ids"])
    out = torch.zeros(S, K, dtype=dtype)
    rows = torch.repeat_interleave(torch.arange(S), shard["row_ptr"][1:] - shard["row_ptr"][:-1])
    out[rows, shard["col"].long()] = values.to(dtype)
    return out


def dense_counts(shard):
    return _dense(shard, shard["counts"], torch.long)


def dense_sums(shard):
    return _dense(shard, shard["sums"], torch.float64)


def validate_run(out_dir, expected_conditions, expected_languages, expected_sentences=None):
    root = Path(out_dir) / "association"; m = json.loads((root / "manifest.json").read_text())
    if not m["complete"]:
        raise CollectError(f"the run is incomplete: missing conditions {m['missing_conditions']}, missing shards {len(m['missing_shards'])}")
    if m["languages"] != list(expected_languages):
        raise CollectError(f"languages {m['languages']} differ from the expected roster {list(expected_languages)}")
    if set(m["conditions"]) != set(expected_conditions):
        raise CollectError(f"conditions {m['conditions']} differ from the expected {list(expected_conditions)}")
    for s, f in m["profiles"].items():
        if _sha256(root / f["file"]) != f["sha256"]:
            raise CollectError(f"hash mismatch for {f['file']}")
    for s, per_cond in m["shards"].items():
        for c, per_lang in per_cond.items():
            for l, x in per_lang.items():
                if _sha256(root / x["file"]) != x["sha256"]:
                    raise CollectError(f"hash mismatch for {x['file']}")
                if expected_sentences is not None and x["n_sentences"] != expected_sentences[l]:
                    raise CollectError(f"sentence count of {s}/{c}/{l} is {x['n_sentences']}, expected {expected_sentences[l]}")
    return m


def make_sink(out_dir, regime, languages, n_feat, bits, baseline_dir, equal_budget, subset=False, sentence_streams=None, sums_dtype="float64"):
    """The sink of a real run: mask regime only; the baseline directory (original mask/eq) supplies the frozen feature union (selective plus matched controls) for the sentence rows."""
    if regime != "mask":
        raise CollectError("association collection is defined for the mask regime only (D-028)")
    if baseline_dir is None:
        raise CollectError("--association-baseline (the original mask/eq directory) is required with --association-profiles")
    path = Path(baseline_dir) / "selectivity.pt"
    sel = torch.load(path, map_location="cpu", weights_only=True)
    base = list(sel["languages"])
    if (list(languages) != base) and not (subset and set(languages) <= set(base)):
        raise CollectError(f"languages {list(languages)} differ from the baseline roster {base}")
    union = sel["selective"].bool().clone(); union[sel["controls"].long()] = True
    ids = union.nonzero().squeeze(1)
    if int(sel["selective"].shape[0]) != int(n_feat):
        raise CollectError("baseline feature count differs from the run")
    cfg = {"regime": regime, "bits": list(bits), "baseline_selectivity_sha256": _sha256(path), "subset": bool(subset), "equal_budget": bool(equal_budget)}
    streams = ("eq", "all") if equal_budget else ("all",)
    ss = None if sentence_streams is None else tuple(s for s in sentence_streams if s in streams)
    return ProfileSink(out_dir, languages, n_feat, ["FP"] + [condition_name(b) for b in bits], ids, streams, config=cfg, sentence_streams=ss, sums_dtype=sums_dtype)

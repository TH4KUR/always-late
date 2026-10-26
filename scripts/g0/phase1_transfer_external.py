"""The six-language external (Wikipedia) panel of the intervention-transfer study (amendment A11; panel freeze `freeze_external.json`, built by phase1_transfer_panel.py). Loads and verifies the passages (text file and
every passage by sha256, rank, split and counts; the token count of every passage is checked against the freeze once the tokenizer is available), and builds the external CORE of the feature-level run: the 404 features of
the main core with contexts chosen by the F5 rule (the M passages of the largest full-precision activation among those where the feature is positive) on the external EVALUATION passages. The selection uses
full-precision activations only; the core and its hashes are written before the quantized run, which refuses a core that differs from the record."""
import hashlib
import json
from pathlib import Path


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def load_passages(freeze, text_dir, split="evaluation"):
    """{"codes": languages in the order of the freeze that are complete, "texts": {code: [text in rank order]}, "n_tokens": {code: [frozen token count]}, "missing": languages recorded as missing}. Verifies the sha256 of
    the text file, of every passage, and its article id, rank and split against the freeze; the number of passages of the split must equal the freeze's count. A missing language is listed, never replaced."""
    codes, texts, n_tokens, missing = [], {}, {}, []
    for code, rec in freeze["languages"].items():
        if rec.get("missing"):
            missing.append(code); continue
        path = Path(text_dir) / rec["text_file"]["file"]; body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != rec["text_file"]["sha256"]:
            raise ValueError(f"{code}: the sha256 of {path.name} differs from the freeze")
        rows = [json.loads(line) for line in body.decode().splitlines() if line.strip()]
        if len(rows) != len(rec["passages"]):
            raise ValueError(f"{code}: {len(rows)} passages in the file, {len(rec['passages'])} in the freeze")
        for row, fr in zip(rows, rec["passages"]):
            if sha256_text(row["text"]) != row["sha256"] or row["sha256"] != fr["sha256"] or str(row["article_id"]) != str(fr["article_id"]) or row["rank"] != fr["rank"] or row["split"] != fr["split"]:
                raise ValueError(f"{code}: passage of rank {fr['rank']} does not match the freeze (text hash, article id, rank or split)")
        keep = [(row, fr) for row, fr in zip(rows, rec["passages"]) if fr["split"] == split]
        if len(keep) != rec[f"n_{split}"]:
            raise ValueError(f"{code}: {len(keep)} {split} passages, the freeze says {rec[f'n_{split}']}")
        keep.sort(key=lambda x: x[1]["rank"]); codes.append(code); texts[code] = [r["text"] for r, _ in keep]; n_tokens[code] = [int(f["n_tokens"]) for _, f in keep]
    return {"codes": codes, "texts": texts, "n_tokens": n_tokens, "missing": missing}


def check_token_counts(tokens, n_tokens):
    """tokens {code: [token id list per passage]} against the frozen counts {code: [n]}: every passage must have exactly the frozen number of tokens (same tokenizer, no special tokens)."""
    for code, lists in tokens.items():
        if len(lists) != len(n_tokens[code]):
            raise ValueError(f"{code}: {len(lists)} passages tokenized, the freeze has a number of {len(n_tokens[code])}")
        bad = [(i, len(t), n) for i, (t, n) in enumerate(zip(lists, n_tokens[code])) if len(t) != n]
        if bad:
            raise ValueError(f"{code}: {len(bad)} passages with another token count than the freeze, first (index, got, frozen): {bad[0]}")


def external_core(core, max_act, M, min_ctx):
    """The external core. core: the main core sample (feature ids, matched direction ids, language indices); max_act [n_passages, n_features] (column k = feature core['feature_ids'][k]): the maximum full-precision
    activation of each feature in each external evaluation passage. Contexts by the F5 rule (`select_contexts`); a feature with fewer than `min_ctx` positive passages is excluded and listed with its count."""
    import phase1_func_f5_run as fr5
    n_pass, n_f = max_act.shape
    if n_f != len(core["feature_ids"]):
        raise ValueError(f"{n_f} columns for {len(core['feature_ids'])} features")
    sel = fr5.select_contexts(max_act, M, min_ctx); keep = [k for k in range(n_f) if not bool(sel["excluded"][k])]
    return {"kind": "intervention-transfer-core-external", "schema": 1, "feature_ids": [int(core["feature_ids"][k]) for k in keep], "direction_ids": [int(core["direction_ids"][k]) for k in keep],
            "feature_language_index": [int(core["feature_language_index"][k]) for k in keep], "languages": list(core["languages"]),
            "contexts": [[int(c) for c in sel["contexts"][k].tolist() if c >= 0] for k in keep],
            "excluded_features": [{"feature": int(core["feature_ids"][k]), "n_active": int(sel["n_active"][k])} for k in range(n_f) if bool(sel["excluded"][k])],
            "parameters": {"M": int(M), "min_ctx": int(min_ctx), "n_passages": int(n_pass), "rule": "F5: the M passages of the largest maximum full-precision activation among those where the feature is positive; ties to the smaller index"},
            "source": {"main_core_feature_ids_sha256": sha256_text(json.dumps([int(x) for x in core["feature_ids"]]))}}


def core_record(core, text):
    """The record that pins an external core: the sha256 of the file text, of the feature ids and of the contexts."""
    return {"file_sha256": sha256_text(text), "feature_ids_sha256": sha256_text(json.dumps(core["feature_ids"])), "contexts_sha256": sha256_text(json.dumps(core["contexts"])), "n_features": len(core["feature_ids"]),
            "n_excluded": len(core["excluded_features"])}


def check_external_core(record, core, text):
    """The core loaded for the quantized run must be the one of the record: file, feature ids, contexts. Every difference is reported in one error."""
    problems = []
    if sha256_text(text) != record["file_sha256"]:
        problems.append("the external core file differs from the record (file sha256)")
    if sha256_text(json.dumps(core["feature_ids"])) != record["feature_ids_sha256"]:
        problems.append("the external core's feature ids differ from the record")
    if sha256_text(json.dumps(core["contexts"])) != record["contexts_sha256"]:
        problems.append("the external core's contexts differ from the record")
    if problems:
        raise ValueError("; ".join(problems))

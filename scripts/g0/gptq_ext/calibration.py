"""The multilingual calibration pool of the GPTQ extension (plan Task 4; audit section 6). GPTQ chooses weights from calibration inputs, so the text must be independent of everything the experiments are evaluated on: FLORES
dev and devtest (every passage containing a normalized FLORES line is dropped) and the six-language Wikipedia panel of the transfer study (its articles, calibration and evaluation passages alike, are excluded by article id).

Source: the pinned Wikimedia Wikipedia snapshot of the external panel (`wikimedia/wikipedia`, config date 20231101, revision b04c8d1c), the first parquet shard of each language's config (`reports/gptq_extension/
wikipedia_first_shards.json`, sha256 from the Hugging Face file listing; the six panel shards agree with `phase1_transfer_panel.SHARDS`). Per language: articles in the order of SHA-256('salt:article id') (salt differs from the
panel's); a passage is the first (at most 128) tokens of an article (Gemma tokenizer, no special tokens), the article must have at least 64 tokens; at least 80 % of the letters must be in the language's script; zho_Hans passages
with a Traditional-only character are dropped; accepted passages are taken in that order until the language has EXACTLY its budget of content tokens (the last passage is cut to fit). A language that cannot reach the budget keeps
what it has and records the shortfall: nothing is ever filled from another language. Sequences are `[BOS] + tokens`; the Hessian statistics later exclude padding only.
Unlike the panel (which cuts at sentence ends and needs a sentence splitter), the first tokens of an article are used as they are: GPTQ needs activation statistics, not sentences, and the panel's splitter does not know the full
stops of Thai, Khmer, Lao or Amharic.

  python3 scripts/g0/gptq_ext/calibration.py --shards-dir <dir> --flores-root <flores200_dataset> --text-dir <dir> --freeze-out <json> --external-freeze reports/intervention_transfer/freeze_external.json --download"""
import argparse
import datetime
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase1_transfer_panel as pp  # noqa: E402

SALT = "gptq-calibration-v1"
BUDGET_TOKENS, MAX_LEN, MIN_TOKENS, MIN_SCRIPT_FRACTION, HEAD_CHARS = 8192, 128, 64, 0.8, 4000
REVISION = pp.REVISION

# FLORES code -> Wikipedia config, ISO 15924 script (the roster's `script`), and the Traditional-only filter of the panel for Mandarin; order = the roster's
LANGS = {code: {"config": cfg, "script": script, "filter_characters": pp.TRADITIONAL_ONLY if code == "zho_Hans" else ""} for code, cfg, script in [
    ("eng_Latn", "en", "Latn"), ("deu_Latn", "de", "Latn"), ("spa_Latn", "es", "Latn"), ("fra_Latn", "fr", "Latn"), ("por_Latn", "pt", "Latn"), ("rus_Cyrl", "ru", "Cyrl"), ("zho_Hans", "zh", "Hans"),
    ("jpn_Jpan", "ja", "Jpan"), ("kor_Hang", "ko", "Hang"), ("hin_Deva", "hi", "Deva"), ("ben_Beng", "bn", "Beng"), ("tam_Taml", "ta", "Taml"), ("tur_Latn", "tr", "Latn"), ("vie_Latn", "vi", "Latn"),
    ("ind_Latn", "id", "Latn"), ("tha_Thai", "th", "Thai"), ("pol_Latn", "pl", "Latn"), ("ukr_Cyrl", "uk", "Cyrl"), ("kan_Knda", "kn", "Knda"), ("tel_Telu", "te", "Telu"), ("yor_Latn", "yo", "Latn"),
    ("sun_Latn", "su", "Latn"), ("amh_Ethi", "am", "Ethi"), ("ibo_Latn", "ig", "Latn"), ("hau_Latn", "ha", "Latn"), ("khm_Khmr", "km", "Khmr"), ("lao_Laoo", "lo", "Laoo")]}
_HAN = ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
_SCRIPT_PREFIX = {"Latn": ("LATIN",), "Cyrl": ("CYRILLIC",), "Hans": _HAN, "Jpan": ("HIRAGANA", "KATAKANA") + _HAN, "Hang": ("HANGUL",), "Deva": ("DEVANAGARI",), "Beng": ("BENGALI",), "Taml": ("TAMIL",),
                  "Knda": ("KANNADA",), "Telu": ("TELUGU",), "Thai": ("THAI",), "Khmr": ("KHMER",), "Laoo": ("LAO",), "Ethi": ("ETHIOPIC",)}
normalize = pp.normalize


def script_fraction(text, script):
    """Share of the letters of `text` that belong to the ISO 15924 `script` (0.0 without letters)."""
    if script not in _SCRIPT_PREFIX:
        raise ValueError(f"unknown script {script}")
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    return sum(unicodedata.name(c, "").startswith(_SCRIPT_PREFIX[script]) for c in letters) / len(letters) if letters else 0.0


def ids_sha256(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode()).hexdigest()


def build_language(rows, cfg, flores_norm, exclude_ids, encode, decode, budget_tokens, salt, max_len, min_tokens, min_script_fraction, head_chars=HEAD_CHARS):
    """The pool of one language from (article id, text) rows. -> {"passages": [{"article_id", "rank", "ids", "n_tokens", "n_used", "sha256"}], "rejected": {reason: count}, "shortfall": int, "tokens": int,
    "n_articles_available": int}. `encode(text) -> token ids` (no special tokens), `decode(ids) -> text`."""
    docs = {}
    for aid, text in rows:
        docs.setdefault(str(aid), text)
    rejected, passages, total = Counter(), [], 0
    for aid in pp.article_order(docs, salt):
        if total >= budget_tokens:
            break
        if aid in exclude_ids:
            rejected["excluded_article"] += 1; continue
        ids = encode(docs[aid].strip()[:head_chars])
        if len(ids) < min_tokens:
            rejected["length"] += 1; continue
        ids = list(ids[:max_len]); text = decode(ids)
        if script_fraction(text, cfg["script"]) < min_script_fraction:
            rejected["script"] += 1; continue
        if cfg["filter_characters"] and any(c in cfg["filter_characters"] for c in text):
            rejected["filter_character"] += 1; continue
        norm = normalize(text)
        if any(f and f in norm for f in flores_norm):
            rejected["flores_duplicate"] += 1; continue
        used = min(len(ids), budget_tokens - total)
        passages.append({"article_id": aid, "rank": len(passages), "ids": ids[:used], "n_tokens": len(ids), "n_used": used, "sha256": ids_sha256(ids[:used])})
        total += used
    return {"passages": passages, "rejected": dict(rejected), "shortfall": max(0, budget_tokens - total), "tokens": total, "n_articles_available": len(docs)}


def external_article_ids(freeze_external):
    """{language code: set of article ids} of every passage (calibration and evaluation) of the external panel; a language recorded as missing has none."""
    return {code: {str(p["article_id"]) for p in rec["passages"]} for code, rec in freeze_external["languages"].items() if not rec.get("missing")}


def pool_sha256(pool):
    """Hash of the pool in its order: languages as listed, passages by rank, article id and token ids."""
    body = [[code, [[p["article_id"], [int(i) for i in p["ids"]]] for p in res["passages"]]] for code, res in pool.items()]
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()


def pool_batches(pool, bos_id, pad_id, batch_size=16, allow_shortfall=False):
    """The calibration batches, in a fixed order (languages as listed, passages by rank, `batch_size` sequences per batch, one language per batch): [{"lang", "article_ids", "tokens" [B, L] long, "mask" [B, L] bool}].
    Every sequence is [BOS] + its tokens, right-padded; `mask` is True for BOS and content and False for padding only (the Hessian statistics exclude padding, not BOS)."""
    short = {c: r["shortfall"] for c, r in pool.items() if r["shortfall"] > 0}
    if short and not allow_shortfall:
        raise ValueError(f"the pool has a shortfall in {short}; settle it before calibrating (never filled from another language)")
    out = []
    for code, res in pool.items():
        ps = res["passages"]
        for i in range(0, len(ps), batch_size):
            seqs = [[bos_id] + list(p["ids"]) for p in ps[i:i + batch_size]]; width = max(len(s) for s in seqs)
            tokens = torch.full((len(seqs), width), pad_id, dtype=torch.long); mask = torch.zeros((len(seqs), width), dtype=torch.bool)
            for r, s in enumerate(seqs):
                tokens[r, :len(s)] = torch.tensor(s); mask[r, :len(s)] = True
            out.append({"lang": code, "article_ids": [p["article_id"] for p in ps[i:i + batch_size]], "tokens": tokens, "mask": mask})
    return out


def write_text_files(pool, text_dir):
    """One jsonl file per language (article id, rank, token ids, hash); -> {code: {"file", "sha256"}}. The text itself is not stored (CC BY-SA text stays where it was downloaded)."""
    text_dir = Path(text_dir); text_dir.mkdir(parents=True, exist_ok=True); files = {}
    for code, res in pool.items():
        fp = text_dir / f"calibration_{code}.jsonl"
        fp.write_text("".join(json.dumps({k: p[k] for k in ("article_id", "rank", "ids", "n_tokens", "n_used", "sha256")}) + "\n" for p in res["passages"]))
        files[code] = {"file": fp.name, "sha256": pp.sha256_file(fp)}
    return files


def load_pool(freeze, text_dir):
    """The pool of a freeze from its token files, verified: file sha256, every passage hash, counts, shortfalls and the pool hash must equal the freeze."""
    pool = {}
    for code, rec in freeze["languages"].items():
        fp = Path(text_dir) / rec["text_file"]["file"]
        if pp.sha256_file(fp) != rec["text_file"]["sha256"]:
            raise ValueError(f"{code}: sha256 of {fp.name} differs from the freeze")
        rows = [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]
        if len(rows) != len(rec["passages"]):
            raise ValueError(f"{code}: {len(rows)} passages in the file, {len(rec['passages'])} in the freeze")
        for row, fr in zip(rows, rec["passages"]):
            if ids_sha256(row["ids"]) != row["sha256"] or row["sha256"] != fr["sha256"] or str(row["article_id"]) != str(fr["article_id"]) or row["n_used"] != len(row["ids"]):
                raise ValueError(f"{code}: passage of rank {fr['rank']} does not match the freeze")
        pool[code] = {"passages": rows, "shortfall": rec["shortfall"], "tokens": sum(r["n_used"] for r in rows)}
    if pool_sha256(pool) != freeze["pool_sha256"]:
        raise ValueError("the pool hash differs from the freeze")
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and freeze the GPTQ calibration pool (run on the Ada login node).")
    ap.add_argument("--shards-dir", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--tokenizer", default="google/gemma-2-2b"); ap.add_argument("--text-dir", required=True)
    ap.add_argument("--freeze-out", required=True); ap.add_argument("--external-freeze", required=True); ap.add_argument("--shards-json", default=str(Path(__file__).resolve().parents[3] / "reports/gptq_extension/wikipedia_first_shards.json"))
    ap.add_argument("--commit", default="unknown"); ap.add_argument("--budget", type=int, default=BUDGET_TOKENS); ap.add_argument("--languages", nargs="*", help="only these FLORES codes (a smoke run); the freeze then says partial")
    ap.add_argument("--download", action="store_true", help="download each pinned shard into --shards-dir and delete it after its language is built (Ada's home has about 3 GB free)")
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    encode = lambda t: tok(t, add_special_tokens=False)["input_ids"]; decode = lambda ids: tok.decode(ids)
    shard_list = json.loads(Path(a.shards_json).read_text()); ext_text = Path(a.external_freeze).read_text(); ext_ids = external_article_ids(json.loads(ext_text))
    codes = a.languages or list(LANGS); pool, shards = {}, {}
    for code in codes:
        cfg = LANGS[code]; sh = shard_list[cfg["config"]]; path = Path(a.shards_dir) / Path(sh["file"]).name
        if a.download:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download("wikimedia/wikipedia", sh["file"], repo_type="dataset", revision=REVISION, local_dir=a.shards_dir))
        digest = pp.sha256_file(path)
        if digest != sh["sha256"]:
            raise ValueError(f"{path} has sha256 {digest}, the pinned shard has {sh['sha256']}")
        shards[cfg["config"]] = {"file": sh["file"], "sha256": sh["sha256"], "verified": True}
        ids = pp.article_order(pp._article_ids(path), SALT); flores = pp._flores_norm(a.flores_root, code)
        for take in (4000, 20000, len(ids)):
            rows = pp._read_rows(path, set(ids[:take]))
            res = build_language(rows, cfg, flores, ext_ids.get(code, set()), encode, decode, a.budget, SALT, MAX_LEN, MIN_TOKENS, MIN_SCRIPT_FRACTION)
            if res["shortfall"] == 0 or take >= len(ids):
                break
        res["n_articles_available"] = len(ids); res["n_flores_lines"] = len(flores); pool[code] = res
        if a.download and not a.keep_shards:
            path.unlink()
        print(code, "tokens", res["tokens"], "passages", len(res["passages"]), "SHORTFALL %d" % res["shortfall"] if res["shortfall"] else "complete", res["rejected"], flush=True)
    files = write_text_files(pool, a.text_dir)
    rec = {"schema": 1, "kind": "gptq-calibration-pool", "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M IST"), "commit": a.commit, "partial": a.languages is not None,
           "corpus": {"dataset": "wikimedia/wikipedia", "config_date": "20231101", "revision": REVISION, "licences": ["cc-by-sa-3.0", "gfdl"]}, "shards": shards,
           "tokenizer": {"name": a.tokenizer, "special_tokens": False}, "parameters": {"salt": SALT, "budget_tokens_per_language": a.budget, "max_len": MAX_LEN, "min_tokens": MIN_TOKENS, "min_script_fraction": MIN_SCRIPT_FRACTION,
                                                                                      "head_chars": HEAD_CHARS, "filter_characters_zho_Hans": pp.TRADITIONAL_ONLY},
           "exclusions": {"external_freeze_sha256": hashlib.sha256(ext_text.encode()).hexdigest(), "external_article_ids": {c: len(v) for c, v in ext_ids.items()}, "flores": "dev and devtest, normalized substring test"},
           "pool_sha256": pool_sha256(pool), "total_tokens": sum(r["tokens"] for r in pool.values()), "complete": all(r["shortfall"] == 0 for r in pool.values()),
           "languages": {code: {"config": LANGS[code]["config"], "n_articles_available": r["n_articles_available"], "n_flores_lines": r["n_flores_lines"], "rejected": r["rejected"], "shortfall": r["shortfall"], "tokens": r["tokens"],
                                "text_file": files[code], "passages": [{k: p[k] for k in ("article_id", "rank", "n_tokens", "n_used", "sha256")} for p in r["passages"]]} for code, r in pool.items()}}
    Path(a.freeze_out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("wrote", a.freeze_out, "total tokens", rec["total_tokens"], "complete" if rec["complete"] else "INCOMPLETE")


if __name__ == "__main__":
    main()

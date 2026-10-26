"""External validation panel of the intervention-transfer study (D-088; ledger reports/intervention_transfer/STATUS.md item X; freeze spec `external_panel`): six languages, 100 calibration and 300 evaluation passages
each, from the pinned Wikimedia Wikipedia snapshot. The pure functions below are tested; `main` reads the parquet shards, the FLORES lines and the Gemma tokenizer on Ada (pyarrow and transformers are imported only
there) and writes the passage texts (NOT committed: CC BY-SA text stays on Ada) and the freeze amendment (article ids, text hashes, token counts, reject counts, shard hashes), which is committed before any outcome
on the panel is read.

Rules: the first parquet shard of each config; articles in the order of SHA-256('salt:article id'); per article the passage is the sentences from the start of the text while the whole stays within 128 tokens
(Gemma tokenizer, no special tokens), at least 64; the passage must be at least 80 % letters of the language's script; a passage with a sentence equal (normalized) to a FLORES dev or devtest line of the language
is dropped; zho_Hans passages with a character of the Traditional-only filter are dropped; the first 100 accepted passages are calibration, the next 300 evaluation; fewer than 400 accepted = the language is
recorded as MISSING (never padded, never replaced). Sentences end at . ! ? and their full-width or Devanagari forms: an approximation (abbreviations split), documented, not tuned.

  python3 scripts/g0/phase1_transfer_panel.py --shards-dir <dir with the six parquet files> --flores-root <flores200_dataset> --tokenizer google/gemma-2-2b --text-dir <dir> --freeze-out <json> --commit <hash>"""
import argparse
import datetime
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SALT = "intervention-transfer-external-v1"
MIN_TOKENS, MAX_TOKENS, MIN_SCRIPT_FRACTION, N_CALIBRATION, N_EVALUATION = 64, 128, 0.8, 100, 300
TRADITIONAL_ONLY = "這個們來說為與對會時學國將後開長關點從還麼種現發裡當問應頭見無動經讓樣實體電機車東書氣門馬歲親邊運進"      # equal to phase1_transfer_freeze.TRADITIONAL_ONLY (a test checks it)

PANEL = {
    "eng_Latn": {"config": "en", "script": "latin", "filter_characters": ""},
    "spa_Latn": {"config": "es", "script": "latin", "filter_characters": ""},
    "rus_Cyrl": {"config": "ru", "script": "cyrillic", "filter_characters": ""},
    "hin_Deva": {"config": "hi", "script": "devanagari", "filter_characters": ""},
    "zho_Hans": {"config": "zh", "script": "han", "filter_characters": TRADITIONAL_ONLY},
    "yor_Latn": {"config": "yo", "script": "latin", "filter_characters": ""},
}
# the first parquet shard of each config at REVISION, with the sha256 that the Hugging Face file listing gives (checked by `main` before any use)
SHARDS = {
    "en": {"file": "20231101.en/train-00000-of-00041.parquet", "sha256": "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677"},
    "es": {"file": "20231101.es/train-00000-of-00013.parquet", "sha256": "905d5162b5b8fa45f50ea8371cd1b7f4c6510b43ab2e5e8f0fd247462d0be1cc"},
    "ru": {"file": "20231101.ru/train-00000-of-00021.parquet", "sha256": "9f8eaf92accb29a4641010655a50d1c8a5880d22020a5c98ea2666a28c7be66a"},
    "hi": {"file": "20231101.hi/train-00000-of-00002.parquet", "sha256": "48001d229cb28d1e74c1f6bac76a1d0ccfc2ffa05c8fb9c2201a5b56108e7c58"},
    "zh": {"file": "20231101.zh/train-00000-of-00006.parquet", "sha256": "853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56"},
    "yo": {"file": "20231101.yo/train-00000-of-00001.parquet", "sha256": "f884b35dfd0204abb76a83eb36cedaabe91267df58c61680436326234d34b43b"},
}

_END = r'(?:[!?。！？।॥]+|\.(?=\s|$|["”’»)\]]))["”’»)\]]*\s*'
_SENT = re.compile(rf".+?{_END}|.+$", re.S)
_FINAL = "!?。！？।॥."
_SCRIPT_PREFIX = {"latin": ("LATIN",), "cyrillic": ("CYRILLIC",), "devanagari": ("DEVANAGARI",), "han": ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")}


def split_sentences(text):
    """Sentences of a text. Lines without any sentence-final punctuation (titles, headings) are dropped; the rest is joined and cut after . ! ? 。 ！ ？ । ॥ (a full stop inside a number does not end a sentence)."""
    lines = [ln.strip() for ln in text.split("\n")]
    body = " ".join(ln for ln in lines if ln and any(c in _FINAL for c in ln))
    return [m.group(0).strip() for m in _SENT.finditer(body) if m.group(0).strip()]


def normalize(s):
    """Case-, width-, spacing- and punctuation-insensitive form of a sentence (letters and digits of any script survive)."""
    return "".join(c for c in unicodedata.normalize("NFKC", s).lower() if unicodedata.category(c)[0] in "LN")


def script_fraction(text, script):
    if script not in _SCRIPT_PREFIX:
        raise ValueError(f"unknown script {script}")
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(unicodedata.name(c, "").startswith(_SCRIPT_PREFIX[script]) for c in letters) / len(letters)


def article_order(ids, salt):
    return sorted((str(i) for i in ids), key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())


def cut_passage(sentences, count_tokens, lo, hi, joiner=" "):
    """The longest run of sentences from the start whose joined text has at most `hi` tokens (whole text counted), or None if the first sentence already exceeds `hi` or the run has fewer than `lo` tokens."""
    cur = []
    for s in sentences:
        if count_tokens(joiner.join(cur + [s])) > hi:
            break
        cur.append(s)
    if not cur:
        return None
    text = joiner.join(cur); n = count_tokens(text)
    return {"text": text, "n_tokens": n, "n_sentences": len(cur)} if n >= lo else None


def build_language(rows, cfg, flores_norm, count_tokens, n_calibration, n_evaluation, min_tokens, max_tokens, salt, min_script_fraction):
    """The panel of one language from (article id, text) rows. The first acceptable passages in the hash order of the article ids; stops when enough are found."""
    docs = {}
    for aid, text in rows:
        docs.setdefault(str(aid), text)
    needed = n_calibration + n_evaluation; passages = []; rejected = Counter(); joiner = "" if cfg["script"] == "han" else " "
    for aid in article_order(docs, salt):
        if len(passages) >= needed:
            break
        sents = split_sentences(docs[aid]); cut = cut_passage(sents, count_tokens, min_tokens, max_tokens, joiner)
        if cut is None:
            rejected["length"] += 1; continue
        if script_fraction(cut["text"], cfg["script"]) < min_script_fraction:
            rejected["script"] += 1; continue
        if cfg["filter_characters"] and any(c in cfg["filter_characters"] for c in cut["text"]):
            rejected["filter_character"] += 1; continue
        if any(normalize(s) in flores_norm for s in sents[:cut["n_sentences"]]):
            rejected["flores_duplicate"] += 1; continue
        rank = len(passages)
        passages.append({"article_id": aid, "rank": rank, "split": "calibration" if rank < n_calibration else "evaluation", "text": cut["text"], "sha256": hashlib.sha256(cut["text"].encode()).hexdigest(),
                         "n_tokens": cut["n_tokens"], "n_sentences": cut["n_sentences"]})
    return {"passages": passages, "rejected": dict(rejected), "needed": needed, "missing": len(passages) < needed, "n_articles_available": len(docs)}


def freeze_record(results, text_files):
    out = {}
    for code, res in results.items():
        ps = res["passages"]
        out[code] = {"config": PANEL[code]["config"] if code in PANEL else None, "needed": res["needed"], "missing": res["missing"], "n_calibration": sum(p["split"] == "calibration" for p in ps),
                     "n_evaluation": sum(p["split"] == "evaluation" for p in ps), "rejected": res["rejected"], "n_articles_available": res.get("n_articles_available"),
                     "text_file": text_files.get(code), "passages": [{k: p[k] for k in ("article_id", "rank", "split", "sha256", "n_tokens", "n_sentences")} for p in ps]}
    return {"languages": out}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _flores_norm(root, code):
    out = set()
    for split in ("dev", "devtest"):
        for ln in (Path(root) / split / f"{code}.{split}").read_text().splitlines():
            out.add(normalize(ln))
    return out


def _read_rows(path, wanted):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path); rows = []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["id", "text"]).to_pydict()
        rows += [(i, x) for i, x in zip(t["id"], t["text"]) if str(i) in wanted]
    return rows


def _article_ids(path):
    import pyarrow.parquet as pq
    return [str(i) for i in pq.ParquetFile(path).read(columns=["id"]).column("id").to_pylist()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and freeze the external panel of the intervention-transfer study (run on Ada).")
    ap.add_argument("--shards-dir", required=True); ap.add_argument("--flores-root", required=True); ap.add_argument("--tokenizer", default="google/gemma-2-2b"); ap.add_argument("--text-dir", required=True)
    ap.add_argument("--freeze-out", required=True); ap.add_argument("--commit", default="unknown")
    ap.add_argument("--download", action="store_true", help="download each pinned shard into --shards-dir (the disk of Ada has about 3 GB free: a shard is deleted after its language is built unless --keep-shards)")
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer); count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    text_dir = Path(a.text_dir); text_dir.mkdir(parents=True, exist_ok=True); results, files, shards = {}, {}, {}
    for code, cfg in PANEL.items():
        sh = SHARDS[cfg["config"]]; path = Path(a.shards_dir) / Path(sh["file"]).name
        if a.download:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download("wikimedia/wikipedia", sh["file"], repo_type="dataset", revision=REVISION, local_dir=a.shards_dir))
        digest = sha256_file(path)
        if digest != sh["sha256"]:
            raise ValueError(f"{path} has sha256 {digest}, the pinned shard has {sh['sha256']}")
        shards[cfg["config"]] = {**sh, "verified": True}
        ids = article_order(_article_ids(path), SALT); flores = _flores_norm(a.flores_root, code)
        for take in (4000, 20000, len(ids)):
            rows = _read_rows(path, set(ids[:take]))
            res = build_language(rows, cfg, flores, count, N_CALIBRATION, N_EVALUATION, MIN_TOKENS, MAX_TOKENS, SALT, MIN_SCRIPT_FRACTION)
            if not res["missing"] or take >= len(ids):
                break
        res["n_articles_available"] = len(ids); results[code] = res
        fp = text_dir / f"passages_{code}.jsonl"
        fp.write_text("".join(json.dumps({k: p[k] for k in ("article_id", "rank", "split", "text", "sha256", "n_tokens")}, ensure_ascii=False) + "\n" for p in res["passages"]))
        files[code] = {"file": fp.name, "sha256": sha256_file(fp)}
        if a.download and not a.keep_shards:
            path.unlink()
        print(code, "passages", len(res["passages"]), "missing" if res["missing"] else "complete", res["rejected"])
    rec = {"schema": 1, "kind": "intervention-transfer-external-panel", "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M IST"), "commit": a.commit,
           "corpus": {"dataset": "wikimedia/wikipedia", "config_date": "20231101", "revision": REVISION, "licences": ["cc-by-sa-3.0", "gfdl"]}, "shards": shards,
           "tokenizer": {"name": a.tokenizer, "special_tokens": False}, "parameters": {"salt": SALT, "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS, "min_script_fraction": MIN_SCRIPT_FRACTION,
                                                                                      "calibration": N_CALIBRATION, "evaluation": N_EVALUATION, "filter_characters_zho_Hans": TRADITIONAL_ONLY},
           **freeze_record(results, files)}
    Path(a.freeze_out).write_text(json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print("wrote", a.freeze_out)


if __name__ == "__main__":
    main()

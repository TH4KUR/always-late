"""Phase 1 data layer (stdlib only): the 27-language roster and the FLORES-200 reader. Decisions D-010 (all sentences per language), D-011 (tier table), A-020, A-022."""
import json
from pathlib import Path

TIERS = ("high", "mid", "low")
REQUIRED = ("code", "name", "tier", "joshi_class", "script", "script_family")


def load_roster(path):
    """Roster rows, validated against the proposal's rules: 27 distinct languages, 9 per tier, at least 4 scripts per tier, Joshi class in 0-5."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["languages"]
    if len(rows) != 27:
        raise ValueError(f"roster has {len(rows)} languages, expected 27")
    if len({r["code"] for r in rows}) != 27:
        raise ValueError("duplicate language code in the roster")
    for r in rows:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"{r.get('code')}: missing fields {missing}")
        if r["tier"] not in TIERS or not isinstance(r["joshi_class"], int) or not 0 <= r["joshi_class"] <= 5:
            raise ValueError(f"{r['code']}: bad tier or Joshi class")
    for t in TIERS:
        langs = [r for r in rows if r["tier"] == t]
        if len(langs) != 9:
            raise ValueError(f"tier {t} has {len(langs)} languages, expected 9")
        if len({r["script"] for r in langs}) < 4:
            raise ValueError(f"tier {t} has fewer than four scripts")
    return rows


def read_flores_split(root, code, split, expect=None):
    """Sentences of one FLORES-200 language file (dev: 997, devtest: 1,012 per language, verified in V-6); a wrong count is an error."""
    path = Path(root) / "flores200_dataset" / split / f"{code}.{split}"
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if expect is not None and len(lines) != expect:
        raise ValueError(f"{path.name}: {len(lines)} sentences, expected {expect}")
    return lines


def text_stats(root, code, split, expect=None):
    """Sentences, characters (Unicode code points, newline excluded) and whitespace-delimited words of one FLORES-200 file: the denominators of token fertility
    (Srivastava 2026, arXiv:2607.24276, section 3: word fertility = tokens per whitespace-delimited word, character fertility = tokens per character; V-11)."""
    lines = read_flores_split(root, code, split, expect)
    return {"sentences": len(lines), "chars": sum(len(l) for l in lines), "words": sum(len(l.split()) for l in lines)}


def text_stats_all(root, codes, splits=("dev", "devtest")):
    return {c: {sp: text_stats(root, c, sp) for sp in splits} for c in codes}


def encode_sentences(tokenizer, sentences):
    """Token ids WITHOUT special tokens (their build_tokens_2d uses add_special_tokens=False; BOS is added by the regime in phase1_lib)."""
    return [list(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]


def build_corpus(root, roster, split, tokenizer, expect=None):
    """-> ({language index: [token-id lists]}, per-language report [{code, sentences, tokens, mean_len, max_len}])."""
    data, report = {}, []
    for i, r in enumerate(roster):
        ids = encode_sentences(tokenizer, read_flores_split(root, r["code"], split, expect))
        data[i] = ids
        lens = [len(s) for s in ids]
        report.append({"code": r["code"], "sentences": len(ids), "tokens": sum(lens), "mean_len": sum(lens) / max(len(lens), 1), "max_len": max(lens) if lens else 0})
    return data, report

"""Task layer of the functional-validation study (D-039, D-040; design `docs/superpowers/specs/2026-10-21-functional-validation-design.md`, section 3). Stdlib only; the tokenizer is injected.

T-MC: Belebele (Bandarkar et al., ACL 2024) four-way multiple choice, split by the FLORES origin of the passage (calibration = dev, evaluation = devtest) so that evaluation items are text on
which the SAE features were never selected. T-LM: sentence ids of FLORES devtest, identical in every language (FLORES is parallel)."""
import hashlib
import json
from pathlib import Path

LETTERS = ("A", "B", "C", "D")
FORMATS = ("letter", "cloze")
_FIELDS = ("link", "question_number", "flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4", "correct_answer_num")


def norm(s):
    """Whitespace-normalised text (runs of whitespace, newlines included, become one space)."""
    return " ".join(str(s).split())


def load_belebele(root, code, expect=900):
    """Items of one language from `<root>/<code>.jsonl`: {"passage_key", "question_number", "passage", "question", "options" [4], "answer" (0-based), "code"}. A wrong item count, a missing field,
    an empty option or an answer number outside 1..4 is an error (never silently repaired)."""
    path = Path(root) / f"{code}.jsonl"
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        r = json.loads(line)
        missing = [k for k in _FIELDS if k not in r]
        if missing:
            raise ValueError(f"{path.name} line {n + 1}: missing fields {missing}")
        opts = [norm(r[f"mc_answer{i}"]) for i in (1, 2, 3, 4)]
        if not all(opts):
            raise ValueError(f"{path.name} line {n + 1}: empty option")
        try:
            ans = int(r["correct_answer_num"])
        except (TypeError, ValueError):
            raise ValueError(f"{path.name} line {n + 1}: bad correct_answer_num {r['correct_answer_num']!r}")
        if not 1 <= ans <= 4:
            raise ValueError(f"{path.name} line {n + 1}: correct_answer_num {ans} outside 1..4")
        items.append({"passage_key": r["link"], "question_number": int(r["question_number"]), "passage": norm(r["flores_passage"]), "question": norm(r["question"]),
                      "options": opts, "answer": ans - 1, "code": code})
    if expect is not None and len(items) != expect:
        raise ValueError(f"{path.name}: {len(items)} items, expected {expect}")
    return items


def _key(it):
    return it["passage_key"] if "passage_key" in it else it["link"]


def _passage(it):
    return norm(it["passage"] if "passage" in it else it["flores_passage"])


def passage_origin(items, dev_lines, devtest_lines):
    """{passage_key: "dev" | "devtest" | "mixed" | "none"}: where the passage text occurs as an exact run of consecutive FLORES lines (joined by one space after whitespace normalisation).
    "mixed" = an exact run exists in both files; "none" = in neither (excluded downstream and counted). Items of one passage must agree on its text."""
    texts = {}
    for it in items:
        k, p = _key(it), _passage(it)
        if texts.setdefault(k, p) != p:
            raise ValueError(f"passage {k!r}: items disagree on the passage text")
    out = {}
    for name, lines in (("dev", dev_lines), ("devtest", devtest_lines)):
        L = [norm(x) for x in lines]
        starts = {}
        for i, s in enumerate(L):
            if s:
                starts.setdefault(s, []).append(i)
        found = set()
        for k, p in texts.items():
            for first, idxs in starts.items():
                if not p.startswith(first):
                    continue
                for i in idxs:
                    cur, j = first, i
                    while cur != p and p.startswith(cur + " ") and j + 1 < len(L) and L[j + 1]:
                        j += 1; cur = cur + " " + L[j]
                    if cur == p:
                        found.add(k); break
                if k in found:
                    break
        out[name] = found
    return {k: ("mixed" if k in out["dev"] and k in out["devtest"] else "dev" if k in out["dev"] else "devtest" if k in out["devtest"] else "none") for k in texts}


def split_items(items, origin):
    """{"cal": dev-origin items, "eval": devtest-origin items, "excluded": mixed or none}; order preserved; a passage without an origin entry is a KeyError."""
    out = {"cal": [], "eval": [], "excluded": []}
    for it in items:
        o = origin[_key(it)]
        out["cal" if o == "dev" else "eval" if o == "devtest" else "excluded"].append(it)
    return out


def lm_ids(n_total, n_pilot, n_eval, seed):
    """Disjoint sorted sentence ids {"pilot", "eval", "reserve"} of one FLORES split, ordered by a hash of (seed, id) so that the pilot ids and the evaluation ids do not change when the
    other size changes; the same ids are used in every language (the sentences are translations of each other)."""
    if n_pilot + n_eval > n_total:
        raise ValueError(f"{n_pilot} + {n_eval} ids requested from {n_total}")
    order = sorted(range(n_total), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return {"pilot": sorted(order[:n_pilot]), "eval": sorted(order[n_pilot:n_pilot + n_eval]), "reserve": sorted(order[n_pilot + n_eval:])}


def format_prompt(it, fmt):
    """Prompt text without BOS. letter: passage, question, four lettered options, `Answer:` (scored on the next token among ' A'..' D'); cloze: passage, question, `Answer:` (each option is
    scored as a continuation)."""
    head = f"Passage: {norm(it['passage'])}\nQuestion: {norm(it['question'])}\n"
    if fmt == "letter":
        return head + "".join(f"{L}. {norm(o)}\n" for L, o in zip(LETTERS, it["options"])) + "Answer:"
    if fmt == "cloze":
        return head + "Answer:"
    raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")


def _suffix(tokenize, prefix_text, prefix_ids, extra):
    full = list(tokenize(prefix_text + extra))
    if full[:len(prefix_ids)] != list(prefix_ids):
        raise ValueError(f"tokenisation of the prompt is not a prefix of the tokenisation with {extra!r} appended")
    return full[len(prefix_ids):]


def encode_mc(tokenize, it, fmt):
    """{"tokens": prompt ids, "answer": index, "cand_ids": four single next-token ids (letter) or "option_ids": four id lists (cloze)}. Option ids are read off the tokenisation of the
    prompt with the option appended, which must extend the prompt's own tokenisation (else ValueError: no silent boundary effects)."""
    text = format_prompt(it, fmt)
    ids = list(tokenize(text))
    out = {"tokens": ids, "answer": int(it["answer"]), "format": fmt}
    if fmt == "letter":
        cands = []
        for L in LETTERS:
            s = _suffix(tokenize, text, ids, f" {L}")
            if len(s) != 1:
                raise ValueError(f"option letter {L!r} is {len(s)} tokens, expected 1")
            cands.append(s[0])
        if len(set(cands)) != 4:
            raise ValueError("the four option letters are not four distinct tokens")
        out["cand_ids"] = cands
    else:
        out["option_ids"] = [_suffix(tokenize, text, ids, " " + norm(o)) for o in it["options"]]
        if not all(out["option_ids"]):
            raise ValueError("an option has no tokens")
    return out


def build_mc_items(tokenize, items, fmt, max_len):
    """(kept encodings, number dropped): items whose prompt (plus, for cloze, the longest option) exceeds max_len tokens are dropped and counted, never truncated."""
    kept, dropped = [], 0
    for it in items:
        e = encode_mc(tokenize, it, fmt)
        n = len(e["tokens"]) + (max(len(o) for o in e["option_ids"]) if fmt == "cloze" else 0)
        if n > max_len:
            dropped += 1
        else:
            kept.append({**e, "passage_key": it.get("passage_key", it.get("link")), "question_number": it.get("question_number")})
    return kept, dropped

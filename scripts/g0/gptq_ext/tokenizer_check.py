"""Tokenizer equality check for the Gemma-2-9B run (plan docs/superpowers/plans/2026-10-28-gemma9b-three-arm-confirmation.md, Task 2): the frozen GPTQ calibration pool is a set of token ids of the Gemma-2-2B tokenizer
(`reports/gptq_extension/calibration_freeze.json`, `tokenizer.name`). It may be used to calibrate Gemma-2-9B only if the tokenizers are the same: same vocabulary (a hash over the sorted token/id pairs), same special
ids, same encodings of probe strings in the scripts of the roster. Any difference raises `TokenizerMismatch` naming the first differing field; the command line exits 3.

  python -m gptq_ext.tokenizer_check --a google/gemma-2-2b --b google/gemma-2-9b      (from scripts/g0; loads with transformers.AutoTokenizer, offline when the files are cached)"""
import argparse
import hashlib
import json
import sys

# one short string per script of the 27-language roster family (Latin, Cyrillic, Devanagari, Thai, Khmer, Han, Ethiopic) and a digit string
PROBES = (
    "The quick brown fox jumps over the lazy dog.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ",
    "ក្រពើពណ៌ត្នោតលោតលើឆ្កែខ្ជិល",
    "敏捷的棕色狐狸跳过了懒狗。一二三",
    "ፈጣኑ ቡናማ ቀበሮ ሰነፍ ውሻ ላይ ዘለለ።",
    "0123456789 3.14159 2026-10-28",
)
FIELDS = ("vocab_size", "special_ids", "vocab_sha256", "probe_ids")


class TokenizerMismatch(Exception):
    """The two tokenizers differ (the message names the first differing field)."""


def fingerprint(tok):
    """-> {"vocab_size", "special_ids" [bos, eos, pad, unk], "vocab_sha256", "probe_ids" [token ids of each probe, no special tokens]} of a tokenizer with `get_vocab`, `encode` and the four `*_token_id` attributes."""
    vocab = tok.get_vocab()
    h = hashlib.sha256()
    for token, i in sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])):
        h.update(f"{i}\t{token}\n".encode("utf-8"))
    return {"vocab_size": len(vocab), "special_ids": [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id], "vocab_sha256": h.hexdigest(),
            "probe_ids": [list(tok.encode(p, add_special_tokens=False)) for p in PROBES]}


def check_same(tok_a, tok_b):
    """-> {"a": fingerprint, "b": fingerprint, "equal": True}; raises TokenizerMismatch naming the first differing field (in the order of FIELDS)."""
    a, b = fingerprint(tok_a), fingerprint(tok_b)
    for field in FIELDS:
        if a[field] != b[field]:
            raise TokenizerMismatch(f"the tokenizers differ in {field}: {json.dumps(a[field])[:200]} against {json.dumps(b[field])[:200]}")
    return {"a": a, "b": b, "equal": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Refuse to reuse the 2B calibration pool for a model whose tokenizer is not the 2B one.")
    ap.add_argument("--a", required=True, help="Hugging Face id or path of the first tokenizer (the one the pool was built with)")
    ap.add_argument("--b", required=True, help="Hugging Face id or path of the second tokenizer")
    ap.add_argument("--out", help="write the two fingerprints here (json)")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    ta, tb = AutoTokenizer.from_pretrained(a.a), AutoTokenizer.from_pretrained(a.b)
    try:
        r = check_same(ta, tb)
    except TokenizerMismatch as e:
        print("TOKENIZER MISMATCH:", e, file=sys.stderr)
        return 3
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print(f"tokenizers equal: vocab {r['a']['vocab_size']}, special ids {r['a']['special_ids']}, vocab sha256 {r['a']['vocab_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

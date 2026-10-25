"""Task 17 part (b) of the functional-validation plan: the prompt cross-check of the Belebele pipeline. Runs the frozen T-MC runner (`phase1_func_run.main_mc`) with the letter prompt of the frozen task module
(variant `frozen`) or with the prompt of the lm-evaluation-harness Belebele task (variant `harness`, source S08 of reports/research_sources.md):

    P: {passage}\\nQ: {question.strip()}\\nA: {option 1}\\nB: {option 2}\\nC: {option 3}\\nD: {option 4}\\nAnswer:

scored on the next token among ' A' to ' D' exactly as the frozen letter format is (the harness scores the continuations ' A' to ' D', one token each, so the arg max is the same). The fields are the ones the
frozen loader returns (already whitespace-normalised, the same in both variants). No frozen file is touched: the only change is `phase1_func_tasks.format_prompt`, replaced for the letter format inside the
`harness` run and restored afterwards, even after an error. Every other argument goes to the frozen runner:

    python phase1_func_promptcheck.py --variant harness --out <dir> --task-arguments-of-main_mc ...   (letter format only)"""
import contextlib

import phase1_func_run as fr
import phase1_func_tasks as ft

VARIANTS = ("frozen", "harness")


def harness_prompt(it):
    """The harness text of one item (letter format)."""
    opts = it["options"]
    if len(opts) != 4:
        raise ValueError(f"an item needs four options, got {len(opts)}")
    return f"P: {it['passage']}\nQ: {it['question'].strip()}\n" + "".join(f"{L}: {o}\n" for L, o in zip("ABCD", opts)) + "Answer:"


@contextlib.contextmanager
def use_variant(variant):
    """Inside the block `ft.format_prompt(item, 'letter')` is the harness prompt when variant is 'harness' (other formats go to the frozen function); the frozen function is restored on exit."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    original = ft.format_prompt
    if variant == "harness":
        def patched(it, fmt):
            return harness_prompt(it) if fmt == "letter" else original(it, fmt)
        ft.format_prompt = patched
    try:
        yield
    finally:
        ft.format_prompt = original


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Task 17 (b): the frozen T-MC runner with the frozen or the harness letter prompt (the remaining arguments go to phase1_func_run.main_mc).", allow_abbrev=False)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    a, rest = ap.parse_known_args(argv)
    if "--formats" in rest:
        i = rest.index("--formats") + 1
        vals = []
        while i < len(rest) and not rest[i].startswith("--"):
            vals.append(rest[i]); i += 1
        if any(v != "letter" for v in vals):
            ap.error("only the letter format is replaced by this wrapper: --formats must be 'letter'")
    with use_variant(a.variant):
        fr.main_mc(rest)


if __name__ == "__main__":
    main()

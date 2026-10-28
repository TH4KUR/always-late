"""Run ONCE on the Ada LOGIN node (it needs internet; the compute nodes of the sbatch may be offline): downloads Gemma-2-9B and the Gemma Scope layer-20 SAE into $HF_HOME and checks they resolve.

    export HF_HOME=$BASE/hf_cache HF_TOKEN=<token with the Gemma licence accepted>
    python scripts/g0/prefetch_gemma9b.py"""
import os
import sys

MODEL = "google/gemma-2-9b"
SAE_REPO = "google/gemma-scope-9b-pt-res"
SAE_RELEASE, SAE_ID = "gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical"


def main():
    if not os.environ.get("HF_HOME"):
        sys.exit("set HF_HOME first (the sbatch uses $BASE/hf_cache)")
    from huggingface_hub import snapshot_download
    print("model ->", snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"]), flush=True)
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cpu")
    sae = sae[0] if isinstance(sae, tuple) else sae
    cfg = sae.cfg
    print("SAE ok: hook", getattr(cfg, "hook_name", None), "d_in", cfg.d_in, "d_sae", cfg.d_sae, flush=True)
    assert cfg.d_in == 3584 and getattr(cfg, "hook_name", "blocks.20.hook_resid_post") == "blocks.20.hook_resid_post"
    print("PREFETCH OK")


if __name__ == "__main__":
    main()

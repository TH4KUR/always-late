"""Prime the persistent Hugging Face cache on Ada (run ONCE on the login node, which has network): the SAE, WikiText-2 and the tokenizer / config files that
phase1_run.py needs, so jobs can run with HF_HUB_OFFLINE=1 (the model shards come from fetch_model.sh). Why: three jobs downloading 10.5 GB each in parallel
stalled for 20+ minutes on 2026-10-17 (the single login-node download ran at ~20 MB/s).
  HF_HOME=/home2/karma/duan-verify/hf_cache HF_TOKEN_PATH=/home2/karma/duan-verify/hf/token env/bin/python scripts/g0/prime_hf_cache.py"""
from datasets import load_dataset
from sae_lens import SAE
from transformers import AutoConfig, AutoTokenizer

sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical", device="cpu")
sae = sae[0] if isinstance(sae, tuple) else sae
print("SAE d_in / d_sae:", sae.cfg.d_in, sae.cfg.d_sae, "hook:", getattr(sae.cfg, "hook_name", None))
for split in ("test", "train"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split); print("wikitext", split, len(ds))
print(AutoConfig.from_pretrained("google/gemma-2-2b").model_type, type(AutoTokenizer.from_pretrained("google/gemma-2-2b")).__name__)

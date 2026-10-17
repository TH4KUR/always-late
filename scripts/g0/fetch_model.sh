#!/bin/bash
# Download the gated model ONCE into a persistent cache on /home2 (login node, resumable), so jobs do not each fetch 10.5 GB through a throttled link.
cd /home2/karma/duan-verify
export HF_TOKEN_PATH=/home2/karma/duan-verify/hf/token HF_HOME=/home2/karma/duan-verify/hf_cache
for attempt in $(seq 1 30); do
  echo "=== attempt $attempt $(date)" >> fetch_model.log
  if env/bin/hf download google/gemma-2-2b model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors config.json generation_config.json model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json >> fetch_model.log 2>&1; then echo "=== MODEL OK $(date)" >> fetch_model.log; exit 0; fi
  echo "=== attempt $attempt failed $(date)" >> fetch_model.log; sleep 20
done
echo "=== GAVE UP $(date)" >> fetch_model.log

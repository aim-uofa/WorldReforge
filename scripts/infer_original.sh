#!/usr/bin/env bash
set -euo pipefail

checkpoint=models/Cosmos3-Nano
manifest=manifests/eval1200.jsonl
data_root=data/reforge-eval
vae=models/Wan2.2_VAE.pth
text_tokenizer=models/Qwen3-VL-8B-Instruct-tokenizer
output="${COSIM_RESULTS_ROOT:-outputs}/nano-original/inference"
gpu=0
batch_size=1

tokenizer_args=()
if [[ -d "$text_tokenizer" ]]; then
  tokenizer_args=(--text-tokenizer "$text_tokenizer")
fi

exec "${PYTHON:-python}" -m worldreforge infer \
  --manifest "$manifest" --data-root "$data_root" \
  --checkpoint "$checkpoint" --tokenizer "$vae" "${tokenizer_args[@]}" \
  --gpu "$gpu" --batch-size "$batch_size" \
  --output "$output" \
  "$@"

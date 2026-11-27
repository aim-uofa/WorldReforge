#!/usr/bin/env bash
set -euo pipefail

checkpoint=models/nano-worldreforge
manifest=manifests/eval1200.jsonl
data_root=data/reforge-eval
vae=models/Wan2.2_VAE.pth
text_tokenizer=models/Qwen3-VL-8B-Instruct-tokenizer
tracker_repo=third_party/co-tracker
tracker_checkpoint=models/cotracker3_offline.pth
output="${COSIM_RESULTS_ROOT:-outputs}/reforge-eval"
gpu=0
batch_size=1
cpu_workers=8

tokenizer_args=()
if [[ -d "$text_tokenizer" ]]; then
  tokenizer_args=(--text-tokenizer "$text_tokenizer")
fi

exec "${PYTHON:-python}" -m worldreforge run \
  --manifest "$manifest" --data-root "$data_root" \
  --checkpoint "$checkpoint" --tokenizer "$vae" "${tokenizer_args[@]}" \
  --tracker-repository "$tracker_repo" --tracker-checkpoint "$tracker_checkpoint" \
  --gpu "$gpu" --batch-size "$batch_size" --cpu-workers "$cpu_workers" \
  --output "$output" \
  "$@"

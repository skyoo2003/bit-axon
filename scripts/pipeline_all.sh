#!/usr/bin/env bash
# Run the Bit-Axon pipeline for small / medium / large configs and stage
# HuggingFace upload folders locally (no push).
#
# Usage:
#   ./scripts/pipeline_all.sh [size]
#
#   size: one of "small", "medium", "large", "all" (default: all)
#
# Env overrides:
#   TOKENIZER   HF tokenizer id/path (default: Qwen/Qwen2.5-3B)
#   SFT_DATA    SFT dataset preset/HF id (default: ultrachat)
#   ORPO_DATA   ORPO dataset preset/HF id (default: ultrafeedback)
#   REPO_OWNER  HF repo owner used in the staged model card (default: skyoo2003)
#   OUTPUT_ROOT Output root (default: pipeline_output)
set -euo pipefail

SIZE="${1:-all}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-3B}"
SFT_DATA="${SFT_DATA:-ultrachat}"
ORPO_DATA="${ORPO_DATA:-ultrafeedback}"
REPO_OWNER="${REPO_OWNER:-skyoo2003}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pipeline_output}"

BIT_AXON="${BIT_AXON:-bit-axon}"

run_size() {
  local label="$1"
  local flag="$2"
  local max_steps="$3"
  local orpo_steps="$4"
  local max_seq_len="$5"
  local sft_limit="$6"
  local orpo_limit="$7"
  local bench_limit="$8"

  local out="${OUTPUT_ROOT}/${label}"
  mkdir -p "${out}"

  echo "============================================================"
  echo "Running Bit-Axon pipeline — ${label}"
  echo "  flag=${flag}  max_steps=${max_steps}  orpo_steps=${orpo_steps}"
  echo "  max_seq_len=${max_seq_len}  sft_limit=${sft_limit}  orpo_limit=${orpo_limit}"
  echo "  bench_limit=${bench_limit}  out=${out}"
  echo "============================================================"

  "${BIT_AXON}" pipeline \
    --output-dir "${out}" \
    --max-steps "${max_steps}" \
    --orpo-steps "${orpo_steps}" \
    --max-seq-len "${max_seq_len}" \
    --lora-rank 8 \
    --batch-size 1 \
    --sft-data "${SFT_DATA}" \
    --sft-limit "${sft_limit}" \
    --orpo-data "${ORPO_DATA}" \
    --orpo-limit "${orpo_limit}" \
    --tokenizer "${TOKENIZER}" \
    "${flag}" \
    --benchmark-limit "${bench_limit}"

  # Stage HF upload folder (no network push)
  "${BIT_AXON}" stage-upload "${out}/final" \
    --repo-id "${REPO_OWNER}/bit-axon-${label}" \
    --tokenizer "${TOKENIZER}"
}

case "${SIZE}" in
  small)
    run_size small  --config-small  30 15 128 50 20 10
    ;;
  medium)
    run_size medium --config-medium 10  5 128 20 10  5
    ;;
  large)
    run_size large  --config-large   5  3 128 10  5  3
    ;;
  all)
    run_size small  --config-small  30 15 128 50 20 10
    run_size medium --config-medium 10  5 128 20 10  5
    run_size large  --config-large   5  3 128 10  5  3
    ;;
  *)
    echo "Unknown size: ${SIZE}" >&2
    echo "Usage: $0 [small|medium|large|all]" >&2
    exit 1
    ;;
esac

echo "Done. Staged upload folders:"
for s in small medium large; do
  if [ -d "${OUTPUT_ROOT}/${s}/final/_hf_upload" ]; then
    echo "  ${OUTPUT_ROOT}/${s}/final/_hf_upload"
  fi
done

#!/usr/bin/env bash
set -euo pipefail

BASELINE_MODEL_ID="${BASELINE_MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
QUANT_MODEL_DIR="${QUANT_MODEL_DIR:-./models/deepseek-r1-llama-8b-llmc}"
TASKS="${TASKS:-wikitext}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
SEED="${SEED:-1234}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-outputs}"
OUT_DIR="$OUT_ROOT/$STAMP"
BASE_OUT="$OUT_DIR/baseline"
QUANT_OUT="$OUT_DIR/quant"
mkdir -p "$BASE_OUT" "$QUANT_OUT"

echo "[bench] Baseline (vLLM) → $BASELINE_MODEL_ID"
lm-eval \
  --model vllm \
  --model_args "pretrained=${BASELINE_MODEL_ID}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --output_path "${BASE_OUT}"

echo
echo "[bench] Quantized (vLLM) → $QUANT_MODEL_DIR"
lm-eval \
  --model vllm \
  --model_args "pretrained=${QUANT_MODEL_DIR}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --output_path "${QUANT_OUT}"

echo
echo "[bench] Results saved under: $OUT_DIR"

if command -v jq >/dev/null 2>&1; then
  base_json=$(ls -1 ${BASE_OUT}/*.json 2>/dev/null | head -n1 || true)
  quant_json=$(ls -1 ${QUANT_OUT}/*.json 2>/dev/null | head -n1 || true)
  if [[ -n "${base_json}" && -n "${quant_json}" ]]; then
    task_key="${TASKS}"
    echo
    echo "[bench] Summary table (${TASKS})"
    printf "%-12s | %-14s | %-14s | %-8s\n" "Metric" "Baseline" "Quantized" "Delta%"
    printf -- "%.0s-" {1..60}; echo
    for metric in bits_per_byte byte_perplexity word_perplexity; do
      b=$(jq -r ".results[\"${task_key}\"][\"${metric}\"].value // .results[\"${task_key}\"][\"${metric}\"] // empty" "${base_json}" 2>/dev/null || true)
      q=$(jq -r ".results[\"${task_key}\"][\"${metric}\"].value // .results[\"${task_key}\"][\"${metric}\"] // empty" "${quant_json}" 2>/dev/null || true)
      if [[ -n "$b" && -n "$q" ]]; then
        delta=$(python - <<PY
b = float("${b}"); q = float("${q}")
print(f"{(q - b) / b * 100:.2f}")
PY
)
        printf "%-12s | %-14s | %-14s | %7s%%\n" "$metric" "$b" "$q" "$delta"
      fi
    done
  else
    echo "[bench] jq found, but result JSON files not located."
  fi
else
  echo "[bench] Install jq to print a summary table (sudo apt-get install -y jq)."
fi



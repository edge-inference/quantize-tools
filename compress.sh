#!/usr/bin/env bash
set -euo pipefail

# Model and output
export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"

if [[ -z "${OUT_DIR:-}" ]]; then
    MODEL_NAME=$(echo "$MODEL_ID" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9.-]/-/g')
    export OUT_DIR="./models/${MODEL_NAME}-llmc"
else
    export OUT_DIR="${OUT_DIR}"
fi
mkdir -p "$OUT_DIR"

# Quant scheme: AWQ only
export SCHEME="${SCHEME:-W4A16}"        # W4A16 is weight-4, act-16 mixed precision suitable for vLLM

# Calibration
export DATASET_NAME="${DATASET_NAME:-wikitext}"
export DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-103-raw-v1}"
export MAX_CALIB_SAMPLES="${MAX_CALIB_SAMPLES:-3072}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
export PAD_TO_MAX_LEN="${PAD_TO_MAX_LEN:-false}"
export SHUFFLE_CALIB="${SHUFFLE_CALIB:-true}"

# Workers
export PREPROC_WORKERS="${PREPROC_WORKERS:-$(nproc)}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYBIN="${PYBIN:-/home/lab/modfi/autoawq/venv/bin/python}"

"$PYBIN" - <<'PY'
import os
from llmcompressor import oneshot
from typing import Any, Dict

model_id = os.environ["MODEL_ID"]
out_dir = os.environ["OUT_DIR"]
scheme = os.environ.get("SCHEME", "W4A16")
dataset = os.environ.get("DATASET_NAME", "wikitext")
num_calib = int(os.environ.get("MAX_CALIB_SAMPLES", "256"))
max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "3072"))
shuffle_calib = os.environ.get("SHUFFLE_CALIB", "true").lower() == "true"
preproc_workers = int(os.environ.get("PREPROC_WORKERS", "8"))
pad_to_max_length = os.environ.get("PAD_TO_MAX_LEN", "false").lower() == "true"

from llmcompressor.modifiers.awq import AWQModifier as Modifier
import datetime

model_name = model_id.split('/')[-1].lower().replace('-', '_')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./logs/{model_name}_{timestamp}"

# Ignore lm_head per common practice when weight-only quantizing
recipe = [Modifier(scheme=scheme, targets="Linear", ignore=["lm_head"])]

oneshot(
    model=model_id,
    dataset=dataset,
    dataset_config_name=os.environ.get("DATASET_CONFIG_NAME"),
    recipe=recipe,
    output_dir=out_dir,
    max_seq_length=max_seq_len,
    num_calibration_samples=num_calib,
    shuffle_calibration_samples=shuffle_calib,
    preprocessing_num_workers=preproc_workers,
    pad_to_max_length=pad_to_max_length,
    log_dir=log_dir,
)

print("Saved llm-compressor checkpoint to:", out_dir)
PY



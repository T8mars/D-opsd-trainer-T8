#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/dopsd_wsl_env.sh"

snapshot_path() {
  local cache_name="$1"
  local model_dir="$HUGGINGFACE_HUB_CACHE/$cache_name"
  local ref_file="$model_dir/refs/main"
  if [[ -f "$ref_file" ]]; then
    local ref
    ref="$(cat "$ref_file")"
    if [[ -d "$model_dir/snapshots/$ref" ]]; then
      printf '%s\n' "$model_dir/snapshots/$ref"
      return 0
    fi
  fi
  find "$model_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

MODEL_PATH="${MODEL_PATH:-$(snapshot_path models--black-forest-labs--FLUX.2-klein-4B)}"
EXP_NAME="${EXP_NAME:-flux2_editing_smoke_1step_lowvram_python}"
RESOLUTION_SCALE="${RESOLUTION_SCALE:-0.25}"
LAUNCHER="${LAUNCHER:-python}"
LOW_VRAM="${LOW_VRAM:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
LAYER_OFFLOAD="${LAYER_OFFLOAD:-0}"
LAYER_OFFLOAD_TRANSFORMER_PERCENT="${LAYER_OFFLOAD_TRANSFORMER_PERCENT:-1.0}"
LAYER_OFFLOAD_TEXT_ENCODER_PERCENT="${LAYER_OFFLOAD_TEXT_ENCODER_PERCENT:-1.0}"
MAIN_PORT="${MAIN_PORT:-60251}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-999}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-999}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
EPOCHS="${EPOCHS:-$((MAX_TRAIN_STEPS + 1))}"
LEARNING_RATE_GEN="${LEARNING_RATE_GEN:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_TEST="${BATCH_SIZE_TEST:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TARGET_RESOLUTION="${TARGET_RESOLUTION:-1024}"
SKIP_INITIAL_SAMPLE="${SKIP_INITIAL_SAMPLE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-../trainer-data/smoke-runs}"
SAMPLE_RESOLUTION_SCALE="${SAMPLE_RESOLUTION_SCALE:-1.0}"
DATA_PATH_TRAIN_JSONL="${DATA_PATH_TRAIN_JSONL:-dataset/interaction/data.jsonl}"
DATA_PATH_TEST_JSONL="${DATA_PATH_TEST_JSONL:-$DATA_PATH_TRAIN_JSONL}"

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT/flux2-klein-edit-self-distill-gt-ref"

TRAIN_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --exp-name "$EXP_NAME"
  --sample-steps "$SAMPLE_STEPS"
  --checkpoint-steps "$CHECKPOINT_STEPS"
  --epochs "$EPOCHS"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --pretrained_model "$MODEL_PATH"
  --num-training-steps 1
  --use-lora 2
  --lora-rank 4
  --lora-alpha 8
  --data-path-train-jsonl "$DATA_PATH_TRAIN_JSONL"
  --data-path-test-jsonl "$DATA_PATH_TEST_JSONL"
  --seed 30
  --mixed-precision bf16
  --batch-size "$BATCH_SIZE"
  --batch-size-test "$BATCH_SIZE_TEST"
  --num-workers 0
  --target-resolution "$TARGET_RESOLUTION"
  --resolution-scale "$RESOLUTION_SCALE"
  --sample-resolution-scale "$SAMPLE_RESOLUTION_SCALE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate-gen "$LEARNING_RATE_GEN"
  --adam-weight-decay 0.0
  --enable-gc
  --vae-dtype bf16
  --ema-decay 0.9999
  --edit-sys-prompt "Note that the last image is the output image I expect to be. Your output must be as similar as the last reference image, based on the first reference image and the editing instructions."
)

if is_truthy "$LOW_VRAM"; then
  TRAIN_ARGS+=(--low-vram)
else
  TRAIN_ARGS+=(--no-low-vram)
fi

if is_truthy "$USE_8BIT_ADAM"; then
  TRAIN_ARGS+=(--use-8bit-adam)
else
  TRAIN_ARGS+=(--no-use-8bit-adam)
fi

if is_truthy "$SKIP_INITIAL_SAMPLE"; then
  TRAIN_ARGS+=(--skip-initial-sample)
else
  TRAIN_ARGS+=(--no-skip-initial-sample)
fi

if is_truthy "$SAVE_SAMPLES"; then
  TRAIN_ARGS+=(--save-samples)
else
  TRAIN_ARGS+=(--no-save-samples)
fi

if is_truthy "$SAVE_CHECKPOINTS"; then
  TRAIN_ARGS+=(--save-checkpoints)
else
  TRAIN_ARGS+=(--no-save-checkpoints)
fi

if [[ "${BLOCK_OFFLOAD:-0}" == "1" || "${BLOCK_OFFLOAD:-}" == "true" ]]; then
  TRAIN_ARGS+=(--block-offload --block-offload-num-blocks "${BLOCK_OFFLOAD_NUM_BLOCKS:-1}")
fi

if is_truthy "$LAYER_OFFLOAD"; then
  TRAIN_ARGS+=(--layer-offload
    --layer-offload-transformer-percent "$LAYER_OFFLOAD_TRANSFORMER_PERCENT"
    --layer-offload-text-encoder-percent "$LAYER_OFFLOAD_TEXT_ENCODER_PERCENT"
  )
fi

if [[ "$LAUNCHER" == "accelerate" ]]; then
  exec accelerate launch \
    --config_file configs/default.yaml \
    --main_process_port "$MAIN_PORT" \
    --num_processes 1 \
    train_dopsd.py \
    "${TRAIN_ARGS[@]}"
fi

exec python train_dopsd.py "${TRAIN_ARGS[@]}"

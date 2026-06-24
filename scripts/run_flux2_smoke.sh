#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/dopsd_wsl_env.sh"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/trainer-data/hf-home/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294}"
EXP_NAME="${EXP_NAME:-flux2_identity_smoke_1step}"
MAIN_PORT="${MAIN_PORT:-60231}"
RESOLUTION_SCALE="${RESOLUTION_SCALE:-0.25}"
LAUNCHER="${LAUNCHER:-python}"
LOW_VRAM="${LOW_VRAM:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-1}"
LAYER_OFFLOAD="${LAYER_OFFLOAD:-0}"
LAYER_OFFLOAD_TRANSFORMER_PERCENT="${LAYER_OFFLOAD_TRANSFORMER_PERCENT:-1.0}"
LAYER_OFFLOAD_TEXT_ENCODER_PERCENT="${LAYER_OFFLOAD_TEXT_ENCODER_PERCENT:-1.0}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
TENSORBOARD="${TENSORBOARD:-1}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tensorboard}"
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
FINAL_SAMPLE_MODE="${FINAL_SAMPLE_MODE:-auto}"
FINAL_SAMPLE_RESOLUTION_SCALE="${FINAL_SAMPLE_RESOLUTION_SCALE:-0.5}"
SAMPLE_RESOLUTION_SCALE="${SAMPLE_RESOLUTION_SCALE:-1.0}"
DATA_PATH_TRAIN_JSONL="${DATA_PATH_TRAIN_JSONL:-dataset/corgi/data.jsonl}"
DATA_PATH_TEST_JSONL="${DATA_PATH_TEST_JSONL:-$DATA_PATH_TRAIN_JSONL}"

is_truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT/flux2-klein_self-distill-edit"

TRAIN_ARGS=(
  --output-dir "$OUTPUT_DIR" \
  --exp-name "$EXP_NAME" \
  --sample-steps "$SAMPLE_STEPS" \
  --checkpoint-steps "$CHECKPOINT_STEPS" \
  --epochs "$EPOCHS" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --pretrained_model "$MODEL_PATH" \
  --num-training-steps 1 \
  --use-lora 2 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --data-path-train-jsonl "$DATA_PATH_TRAIN_JSONL" \
  --data-path-test-jsonl "$DATA_PATH_TEST_JSONL" \
  --seed 30 \
  --mixed-precision bf16 \
  --batch-size "$BATCH_SIZE" \
  --batch-size-test "$BATCH_SIZE_TEST" \
  --num-workers 0 \
  --target-resolution "$TARGET_RESOLUTION" \
  --resolution-scale "$RESOLUTION_SCALE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate-gen "$LEARNING_RATE_GEN" \
  --adam-weight-decay 0.0 \
  --enable-gc \
  --vae-dtype bf16 \
  --final-sample-mode "$FINAL_SAMPLE_MODE" \
  --final-sample-resolution-scale "$FINAL_SAMPLE_RESOLUTION_SCALE" \
  --sample-resolution-scale "$SAMPLE_RESOLUTION_SCALE" \
  --ema-decay 0.9999 \
  --edit-sys-prompt "Keep subject background and composition consistent with the reference image."
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

if is_truthy "$TENSORBOARD"; then
  TRAIN_ARGS+=(--tensorboard --tensorboard-dir "$TENSORBOARD_DIR")
else
  TRAIN_ARGS+=(--no-tensorboard)
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

run_training() {
  if [[ "$LAUNCHER" == "accelerate" ]]; then
    accelerate launch \
      --config_file configs/default.yaml \
      --main_process_port "$MAIN_PORT" \
      --num_processes 1 \
      train_dopsd.py \
      "${TRAIN_ARGS[@]}"
    return $?
  fi

  python train_dopsd.py "${TRAIN_ARGS[@]}"
}

run_deferred_final_sampler() {
  local final_sampler_dir="$OUTPUT_DIR/$EXP_NAME/final_sampler"
  local final_sampler_request=""

  if [[ -d "$final_sampler_dir" ]]; then
    final_sampler_request="$(find "$final_sampler_dir" -maxdepth 1 -type f -name 'request_step_*.json' | sort | tail -n 1)"
  fi

  if [[ -n "$final_sampler_request" ]]; then
    python sample_flux2_final.py --request "$final_sampler_request"
  fi
}

run_training
training_status=$?
if [[ "$training_status" -ne 0 ]]; then
  exit "$training_status"
fi

run_deferred_final_sampler

CONFIG_FILE="configs/default.yaml"
MAIN_PORT=60212
NUM_PROCS=4
PYTHON_SCRIPT="train_dopsd.py"
accelerate launch \
    --config_file      ${CONFIG_FILE} \
    --main_process_port ${MAIN_PORT} \
    --num_processes     ${NUM_PROCS} \
    ${PYTHON_SCRIPT} \
    --deepspeed-config  "configs/z2.json" \
    --output-dir     "exp_results/" \
    --exp-name     "dopsd_editcontext_ema0.9999_onpolicy_4steptrain_4b_corgi_bsz4_lora_lr2e-5" \
    --sample-steps      100 \
    --checkpoint-steps  500 \
    --epochs              3001 \
    --max-train-steps   3001 \
    --pretrained_model   "black-forest-labs/FLUX.2-klein-4B" \
    --num-training-steps  4 \
    --use-lora 2 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --data-path-train-jsonl "dataset/corgi/data.jsonl" \
    --data-path-test-jsonl "dataset/corgi/data.jsonl" \
    --seed   30 \
    --mixed-precision "bf16" \
    --batch-size 1 \
    --batch-size-test 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate-gen 2e-5 \
    --adam-weight-decay 0.0 \
    --enable-gc \
    --vae-dtype "bf16" \
    --ema-decay 0.9999 \
    --edit-sys-prompt "Make the main subject (facial/ip details), background, and overall composition consistent with the reference image." \






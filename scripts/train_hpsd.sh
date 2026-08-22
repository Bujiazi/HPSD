CONFIG_FILE="configs/default.yaml"
DEEPSPEED_CONFIG="configs/z2.json"
MAIN_PORT=${MAIN_PORT:-60232}
NUM_PROCS=${NUM_PROCS:-8}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-5000}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}

accelerate launch \
  --config_file "${CONFIG_FILE}" \
  --main_process_port "${MAIN_PORT}" \
  --num_processes "${NUM_PROCS}" \
  train_hpsd.py \
  --deepspeed-config "${DEEPSPEED_CONFIG}" \
  --output-dir "exp_results" \
  --exp-name "HPSD_WAN2.2" \
  --wan-model-id "pretrained_models/WAN2.2-TI2V" \
  --prompt-file "data/video_prompts.txt" \
  --max-train-samples "${TRAIN_SAMPLES}" \
  --shuffle-prompts \
  --prompt-shuffle-seed "${SHUFFLE_SEED}" \
  --height 704 \
  --width 1280 \
  --num-frames 33 \
  --teacher-inference-steps 50 \
  --teacher-guidance-scale 5.0 \
  --qwen-model-path "pretrained_models/Qwen3.6-27B" \
  --zimage-model-path "pretrained_models/Z-Image-Turbo" \
  --first-frame-cache-dir "cached_data" \
  --zimage-steps 9 \
  --zimage-guidance-scale 0.0 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --inference-steps 50 \
  --anchor-indices "0 5 15 25 30 40" \
  --sub-traj-length-list "3 3 3 3 3 3" \
  --train-guidance-scale 1.0 \
  --save-sample-every-steps 10 \
  --sample-guidance-scale 5.0 \
  --mixed-precision bf16 \
  --transformer-dtype bf16 \
  --vae-dtype fp32 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --ema-decay 0.9999 \
  --adam-weight-decay 0.0 \
  --checkpoint-steps 100 \
  --max-train-steps 500 \
  --enable-gc

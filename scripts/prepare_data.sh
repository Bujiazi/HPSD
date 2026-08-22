set -e

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
TOTAL=${TOTAL:-5000}
SHUFFLE_SEED=${SHUFFLE_SEED:-42}
PER_SHARD=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "Preparing ${TOTAL} randomly-sampled prompts on ${NUM_GPUS} GPU(s), ${PER_SHARD} per shard (seed=${SHUFFLE_SEED})"

pids=()

cleanup() {
  echo ""
  echo "Interrupted, killing background shards..."
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

for i in $(seq 0 $((NUM_GPUS - 1))); do
  start=$(( i * PER_SHARD ))
  if [ "$start" -ge "$TOTAL" ]; then
    break
  fi
  count=$PER_SHARD
  remaining=$(( TOTAL - start ))
  if [ "$count" -gt "$remaining" ]; then
    count=$remaining
  fi
  echo "[shard ${i}] prompts [${start}, $((start + count))) -> GPU ${i}"
  CUDA_VISIBLE_DEVICES=${i} python prepare_hpsd_data.py \
    --prompt-file "data/video_prompts.txt" \
    --prompt-start "${start}" \
    --max-train-samples "${count}" \
    --shuffle-prompts \
    --prompt-shuffle-seed "${SHUFFLE_SEED}" \
    --height 704 \
    --width 1280 \
    --qwen-model-path "pretrained_models/Qwen3.6-27B" \
    --zimage-model-path "pretrained_models/Z-Image-Turbo" \
    --first-frame-cache-dir "cached_data" \
    --zimage-steps 9 \
    --zimage-guidance-scale 0.0 \
    --first-frame-seed 42 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "${pid}" || fail=1
done

trap - INT TERM

if [ "$fail" -ne 0 ]; then
  echo "ERROR: some preparation shards failed; see logs above." >&2
  exit 1
fi
echo "All ${#pids[@]} shard(s) finished."

# One-click download of all pretrained checkpoints into pretrained_models/.
#
# Each model is downloaded with --local-dir (flat layout, no snapshots/ nesting)
# so the resulting folder is directly loadable by from_pretrained / pipelines.
# Existing (non-empty) target folders are skipped, so re-running is safe and
# resume-friendly.
#
# Usage:
#   bash scripts/download_pretrained.sh
set -e

cd "$(dirname "$0")/.."   # project root (HPSD/)

DST="pretrained_models"
mkdir -p "${DST}"

# model -> [repo_id, target_subfolder]
MODELS=(
  "Wan-AI/Wan2.2-TI2V-5B-Diffusers|WAN2.2-TI2V"
  "Qwen/Qwen3.6-27B|Qwen3.6-27B"
  "Tongyi-MAI/Z-Image-Turbo|Z-Image-Turbo"
)

# Check huggingface-cli availability (prefer the python entrypoint).
HF_CLI=""
if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
elif command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif python -c "import huggingface_hub" >/dev/null 2>&1; then
  HF_CLI="python -m huggingface_hub.commands.huggingface_cli"
fi

if [ -z "${HF_CLI}" ]; then
  echo "ERROR: huggingface-cli not found. Install it with:  pip install -U huggingface_hub" >&2
  exit 1
fi

for entry in "${MODELS[@]}"; do
  repo_id="${entry%%|*}"
  sub="${entry##*|}"
  target="${DST}/${sub}"

  if [ -e "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
    echo "[skip] ${target} already exists and is non-empty"
    continue
  fi

  echo "[download] ${repo_id} -> ${target}"
  # shellcheck disable=SC2086
  ${HF_CLI} download "${repo_id}" --local-dir "${target}"
done

echo ""
echo "All pretrained checkpoints are ready under ${DST}/:"
ls -la "${DST}"

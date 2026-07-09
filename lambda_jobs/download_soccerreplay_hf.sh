set -euo pipefail


DATASET_ID="${DATASET_ID:-Homie0609/SoccerReplay-1988}"
OUT_DIR="${OUT_DIR:-/workspace/data/SoccerReplay-1988/hf}"
INCLUDE_PATTERN="${INCLUDE_PATTERN:-*.zip}"

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_TOKEN is required for this gated dataset." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "Downloading $DATASET_ID to $OUT_DIR"
echo "Include pattern: $INCLUDE_PATTERN"

huggingface-cli download "$DATASET_ID" \
  --repo-type dataset \
  --local-dir "$OUT_DIR" \
  --include "$INCLUDE_PATTERN"

if [[ "${UNZIP_AFTER_DOWNLOAD:-0}" == "1" ]]; then
  find "$OUT_DIR" -name '*.zip' -print0 | while IFS= read -r -d '' zipfile; do
    dest="${zipfile%.zip}"
    mkdir -p "$dest"
    unzip -n "$zipfile" -d "$dest"
  done
fi

if [[ -n "${VIDEO_DOWNLOAD_CMD:-}" ]]; then
  echo "Running external video download command from VIDEO_DOWNLOAD_CMD"
  bash -lc "$VIDEO_DOWNLOAD_CMD"
else
  cat <<'MSG'
HF download complete.

Important: SoccerReplay-1988 full-game videos may not be in the HF repo. The
dataset card says the authors provide the video storage link separately after
NDA/access approval. Once you have that link, set VIDEO_DOWNLOAD_CMD to a wget,
aria2c, rclone, or provider-specific command and rerun this script.
MSG
fi

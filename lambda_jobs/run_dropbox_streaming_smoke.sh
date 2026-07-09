set -euo pipefail


CONFIG="${CONFIG:-configs/lambda_smoke.yaml}"
MANIFEST="${MANIFEST:-manifests/dropbox_full_dataset_manifest.csv}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MAX_GAMES="${MAX_GAMES:-1}"
MAX_EVENTS_PER_GAME="${MAX_EVENTS_PER_GAME:-4}"
DEVICE="${DEVICE:-cuda}"
CLEANUP_FRAMES="${CLEANUP_FRAMES:-1}"

python3 scripts/build_dropbox_dataset_manifest.py \
  --annotation-root "${ANNOTATION_ROOT:-/home/ubuntu/Soccer_Data/annotations}" \
  --out-csv "$MANIFEST"

args=(
  python3 scripts/run_dropbox_streaming_shard.py
  --dataset-manifest "$MANIFEST" \
  --config "$CONFIG" \
  --shard-id "$SHARD_ID" \
  --num-shards "$NUM_SHARDS" \
  --max-games "$MAX_GAMES" \
  --device "$DEVICE" \
  --resize-frames "${RESIZE_FRAMES:-0}"
)

if [[ "$MAX_EVENTS_PER_GAME" != "0" ]]; then
  args+=(--max-events-per-game "$MAX_EVENTS_PER_GAME")
fi

if [[ "$CLEANUP_FRAMES" == "1" ]]; then
  args+=(--cleanup-frames)
fi

"${args[@]}"

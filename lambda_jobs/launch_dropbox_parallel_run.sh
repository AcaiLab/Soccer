set -euo pipefail


PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONFIG="${CONFIG:-configs/lambda_smoke.yaml}"
MANIFEST="${MANIFEST:-manifests/dropbox_full_dataset_manifest.csv}"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-/home/ubuntu/Soccer_Data/annotations}"
RUN_NAME="${RUN_NAME:-full_stream_run_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/${RUN_NAME}}"
NUM_SHARDS="${NUM_SHARDS:-8}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-$((NUM_SHARDS - 1))}"
RESIZE_FRAMES="${RESIZE_FRAMES:-160}"
SPLIT="${SPLIT:-all}"

cd "$PROJECT_ROOT"

set -a
source "${PROJECT_ROOT}/.env"
set +a

python3 scripts/build_dropbox_dataset_manifest.py \
  --annotation-root "$ANNOTATION_ROOT" \
  --out-csv "$MANIFEST"

mkdir -p "$OUT_ROOT"
cat > "${OUT_ROOT}/launch_config.json" <<JSON
{
  "run_name": "$RUN_NAME",
  "out_root": "$OUT_ROOT",
  "manifest": "$MANIFEST",
  "config": "$CONFIG",
  "num_shards": $NUM_SHARDS,
  "start_shard": $START_SHARD,
  "end_shard": $END_SHARD,
  "resize_frames": $RESIZE_FRAMES,
  "split": "$SPLIT"
}
JSON

for shard_id in $(seq "$START_SHARD" "$END_SHARD"); do
  gpu=$((shard_id % 8))
  shard_dir="${OUT_ROOT}/shard_$(printf '%04d' "$shard_id")"
  mkdir -p "$shard_dir"
  echo "Launching shard ${shard_id}/${NUM_SHARDS} on visible GPU ${gpu}"
  nohup bash -lc "
    cd '${PROJECT_ROOT}'
    set -a
    source '${PROJECT_ROOT}/.env'
    set +a
    CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/run_dropbox_streaming_shard.py \
      --dataset-manifest '${MANIFEST}' \
      --config '${CONFIG}' \
      --out-root '${OUT_ROOT}' \
      --shard-id '${shard_id}' \
      --num-shards '${NUM_SHARDS}' \
      --split '${SPLIT}' \
      --tmp-video-root '${TMP_VIDEO_BASE:-/home/ubuntu/Soccer_Data/dropbox_tmp_videos_${RUN_NAME}}/shard_$(printf '%04d' "$shard_id")' \
      --resize-frames '${RESIZE_FRAMES}' \
      --device cuda \
      --cleanup-frames \
      --force
  " > "${shard_dir}/worker.out" 2>&1 &
  echo $! > "${shard_dir}/worker.pid"
done

echo "Launched shards ${START_SHARD}-${END_SHARD} into ${OUT_ROOT}"
echo "Monitor with:"
echo "  tail -f ${OUT_ROOT}/shard_0000/worker.out"
echo "  nvidia-smi"

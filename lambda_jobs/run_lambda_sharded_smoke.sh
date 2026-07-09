set -euo pipefail

CONFIG="${CONFIG:-configs/lambda_smoke.yaml}"
SHARDS="${SHARDS:-2}"
DEVICE="${DEVICE:-cuda}"

python scripts/build_window_manifest.py --config "$CONFIG"
python scripts/shard_manifest.py \
  --manifest manifests/local_windows.jsonl \
  --num-shards "$SHARDS"

for shard in $(seq 0 $((SHARDS - 1))); do
  python scripts/run_shard_pipeline.py \
    --config "$CONFIG" \
    --shard-id "$shard" \
    --device "$DEVICE"
done

python scripts/merge_shard_outputs.py \
  --shard-root shards \
  --out features/merged_shard_visual_embeddings.npz

python scripts/build_retrieval_memory.py \
  --config "$CONFIG" \
  --embeddings features/merged_shard_visual_embeddings.npz \
  --out retrieval/merged_shard_memory_mean_std.npz

python scripts/evaluate_retrieval.py \
  --memory retrieval/merged_shard_memory_mean_std.npz

echo "Lambda sharded smoke run complete."

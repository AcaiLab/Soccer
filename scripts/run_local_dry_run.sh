set -euo pipefail

CONFIG="${1:-configs/local_dry_run.yaml}"

python scripts/build_window_manifest.py --config "$CONFIG"
python scripts/extract_visual_embeddings.py --config "$CONFIG"
python scripts/build_retrieval_memory.py --config "$CONFIG" --feature-mode mean_std
python scripts/generate_audience_commentary.py --config "$CONFIG" --max-windows 10
python scripts/build_content_plans.py --config "$CONFIG" --max-windows 10
python scripts/extract_basic_visual_cues.py --config "$CONFIG"
python scripts/merge_visual_facts_into_plans.py
python scripts/generate_natural_commentary.py \
  --config "$CONFIG" \
  --plans outputs/content_plans_with_visual_facts.jsonl

echo "Local dry run complete."

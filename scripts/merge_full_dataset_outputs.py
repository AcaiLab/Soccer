import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pipeline_utils import ensure_dir, repo_root, write_json


RUN_NAMES = [
    "full_stream_run_v100_20260619_refresh_8shards",
    "full_stream_run_v100_20260619_recovery_failed_games",
]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_visual_facts(plan: dict[str, Any]) -> dict[str, Any]:
    visual = plan.get("visual_facts") or {}
    camera = visual.get("camera") or {}
    scene = visual.get("scene") or {}
    field = visual.get("field") or {}
    objects = visual.get("objects") or {}
    return {
        "camera_shot_type": camera.get("shot_type"),
        "field_visible": camera.get("field_visible"),
        "field_visible_ratio": camera.get("field_visible_ratio"),
        "likely_graphic_overlay": camera.get("likely_graphic_overlay"),
        "likely_replay": camera.get("likely_replay"),
        "motion_intensity": scene.get("motion_intensity"),
        "motion_intensity_score": scene.get("motion_intensity_score"),
        "shot_change_rate": scene.get("shot_change_rate"),
        "brightness": scene.get("brightness"),
        "contrast": scene.get("contrast"),
        "edge_density": scene.get("edge_density"),
        "field_zone": field.get("zone"),
        "field_zone_confidence": field.get("confidence"),
        "ball_visible": objects.get("ball_visible"),
        "referee_visible": objects.get("referee_visible"),
        "card_visible": objects.get("card_visible"),
        "substitution_board_visible": objects.get("substitution_board_visible"),
        "player_count_estimate": objects.get("player_count_estimate"),
    }


def compact_retrieval(plan: dict[str, Any]) -> dict[str, Any]:
    retrieval = plan.get("retrieval") or {}
    neighbors = retrieval.get("neighbors") or []
    top = neighbors[0] if neighbors else {}
    return {
        "retrieval_neighbor_count": retrieval.get("neighbor_count", len(neighbors)),
        "retrieval_same_label_neighbors": retrieval.get("same_label_neighbors", 0),
        "retrieval_top_similarity": top.get("similarity"),
        "retrieval_top_event_label": top.get("event_label"),
        "retrieval_top_game_label": top.get("game_label"),
    }


def merge_shard(run_name: str, shard_dir: Path, out_fh: gzip.GzipFile) -> dict[str, Any]:
    windows_path = shard_dir / "windows.jsonl"
    plans_path = shard_dir / "content_plans_with_visual_facts.jsonl"
    natural_path = shard_dir / "natural_commentary_with_visual_facts.jsonl"
    summary_path = shard_dir / "extraction_summary.json"
    shard_name = shard_dir.name

    windows = {row["window_id"]: row for row in iter_jsonl(windows_path)}
    plans = {
        (row["window_id"], row["audience"]): row
        for row in iter_jsonl(plans_path)
    }

    counts = Counter()
    for row in iter_jsonl(natural_path):
        key = (row["window_id"], row["audience"])
        window = windows.get(row["window_id"], {})
        plan = plans.get(key) or {}
        content_plan = row.get("content_plan") or {}
        facts = plan.get("facts") or content_plan.get("facts") or {}
        merged = {
            "run_name": run_name,
            "shard": shard_name,
            "window_id": row.get("window_id"),
            "game_label": row.get("game_label"),
            "event_label": row.get("event_label"),
            "audience": row.get("audience"),
            "commentary": row.get("commentary"),
            "timestamp": window.get("timestamp"),
            "half": window.get("half"),
            "commentary_reference": window.get("commentary_reference"),
            "frame_count": window.get("frame_count"),
            "sampled_frame_count": window.get("sampled_frame_count"),
            "audience_goal": plan.get("audience_goal"),
            "style": plan.get("style"),
            "must_include": plan.get("must_include") or content_plan.get("must_include"),
            "avoid": plan.get("avoid") or content_plan.get("avoid"),
            "plain_event": facts.get("plain_event"),
            "rule_context": facts.get("rule_context"),
            "next_state": facts.get("next_state"),
            "tactical_context": facts.get("tactical_context"),
            "visual_state": facts.get("visual_state"),
            **compact_retrieval(plan or content_plan),
            **compact_visual_facts(plan),
        }
        out_fh.write((json.dumps(merged, ensure_ascii=True) + "\n").encode("utf-8"))
        counts["rows"] += 1
        counts[f"audience:{row.get('audience')}"] += 1
        counts[f"event:{row.get('event_label')}"] += 1

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts["games_done"] = len(summary.get("games_done", []))
        counts["failures"] = len(summary.get("failures", []))
        counts["windows"] = summary.get("windows", len(windows))
    else:
        counts["windows"] = len(windows)
    return dict(counts)


def write_sample_csv(jsonl_gz: Path, out_csv: Path, limit: int) -> None:
    with gzip.open(jsonl_gz, "rt", encoding="utf-8") as f:
        first_rows = []
        for line in f:
            if not line.strip():
                continue
            first_rows.append(json.loads(line))
            if len(first_rows) >= limit:
                break
    if not first_rows:
        return
    fieldnames = list(first_rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(first_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("lambda_results/full_dataset"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/full_dataset_20260625"),
    )
    parser.add_argument("--sample-rows", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    input_root = args.input_root if args.input_root.is_absolute() else base / args.input_root
    out_dir = ensure_dir(args.out_dir if args.out_dir.is_absolute() else base / args.out_dir)
    out_jsonl = out_dir / "merged_commentary.jsonl.gz"
    out_sample = out_dir / "merged_commentary_sample.csv"
    out_summary = out_dir / "merge_summary.json"

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "out_jsonl": str(out_jsonl),
        "runs": {},
        "totals": Counter(),
    }
    with gzip.open(out_jsonl, "wb", compresslevel=6) as out_fh:
        for run_name in RUN_NAMES:
            run_dir = input_root / run_name
            run_counts = Counter()
            for shard_dir in sorted(run_dir.glob("shard_*")):
                shard_counts = merge_shard(run_name, shard_dir, out_fh)
                summary["runs"].setdefault(run_name, {})[shard_dir.name] = shard_counts
                run_counts.update(shard_counts)
            summary["runs"][run_name]["_totals"] = dict(run_counts)
            summary["totals"].update(run_counts)

    summary["totals"] = dict(summary["totals"])
    write_sample_csv(out_jsonl, out_sample, args.sample_rows)
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

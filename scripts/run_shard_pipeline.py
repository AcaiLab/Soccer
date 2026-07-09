import argparse
import json
import subprocess
import time
from pathlib import Path

from pipeline_utils import ensure_dir, write_json


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def maybe_run(name: str, output: Path, cmd: list[str], log_path: Path, force: bool) -> str:
    if output.exists() and output.stat().st_size > 0 and not force:
        return "skipped_existing"
    run(cmd, log_path)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"{name} did not create expected output: {output}")
    return "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lambda_smoke.yaml"))
    parser.add_argument("--shard-root", type=Path, default=Path("shards"))
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    shard_dir = ensure_dir(args.shard_root / f"shard_{args.shard_id:04d}")
    manifest = shard_dir / "windows.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Shard manifest not found: {manifest}")
    outputs = {
        "embeddings": shard_dir / "visual_embeddings.npz",
        "basic_cues": shard_dir / "basic_visual_cues.jsonl",
        "plans": shard_dir / "content_plans.jsonl",
        "plans_visual": shard_dir / "content_plans_with_visual_facts.jsonl",
        "natural": shard_dir / "natural_commentary_with_visual_facts.jsonl",
    }
    log_path = shard_dir / "pipeline.log"
    statuses = {}
    statuses["extract_visual_embeddings"] = maybe_run(
        "extract_visual_embeddings",
        outputs["embeddings"],
        [
            "python",
            "scripts/extract_visual_embeddings.py",
            "--config",
            str(args.config),
            "--manifest",
            str(manifest),
            "--out",
            str(outputs["embeddings"]),
            "--device",
            args.device,
        ],
        log_path,
        args.force,
    )
    memory = shard_dir / "memory_mean_std.npz"
    statuses["build_retrieval_memory"] = maybe_run(
        "build_retrieval_memory",
        memory,
        [
            "python",
            "scripts/build_retrieval_memory.py",
            "--config",
            str(args.config),
            "--embeddings",
            str(outputs["embeddings"]),
            "--out",
            str(memory),
            "--feature-mode",
            "mean_std",
        ],
        log_path,
        args.force,
    )
    statuses["extract_basic_visual_cues"] = maybe_run(
        "extract_basic_visual_cues",
        outputs["basic_cues"],
        [
            "python",
            "scripts/extract_basic_visual_cues.py",
            "--config",
            str(args.config),
            "--manifest",
            str(manifest),
            "--out",
            str(outputs["basic_cues"]),
        ],
        log_path,
        args.force,
    )
    statuses["build_content_plans"] = maybe_run(
        "build_content_plans",
        outputs["plans"],
        [
            "python",
            "scripts/build_content_plans.py",
            "--config",
            str(args.config),
            "--manifest",
            str(manifest),
            "--memory",
            str(memory),
            "--out",
            str(outputs["plans"]),
        ],
        log_path,
        args.force,
    )
    statuses["merge_visual_facts_into_plans"] = maybe_run(
        "merge_visual_facts_into_plans",
        outputs["plans_visual"],
        [
            "python",
            "scripts/merge_visual_facts_into_plans.py",
            "--plans",
            str(outputs["plans"]),
            "--basic-cues",
            str(outputs["basic_cues"]),
            "--out",
            str(outputs["plans_visual"]),
        ],
        log_path,
        args.force,
    )
    statuses["generate_natural_commentary"] = maybe_run(
        "generate_natural_commentary",
        outputs["natural"],
        [
            "python",
            "scripts/generate_natural_commentary.py",
            "--config",
            str(args.config),
            "--plans",
            str(outputs["plans_visual"]),
            "--out",
            str(outputs["natural"]),
        ],
        log_path,
        args.force,
    )
    summary = {
        "shard_id": args.shard_id,
        "manifest": str(manifest),
        "runtime_sec": round(time.time() - start, 3),
        "statuses": statuses,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "memory": str(memory),
        "log": str(log_path),
    }
    write_json(shard_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

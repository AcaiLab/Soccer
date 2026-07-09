import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from pipeline_utils import load_config, repo_root


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_sampling_config(base_config: Path, samples: int, out_path: Path) -> None:
    cfg = load_config(base_config)
    cfg["windows"]["sampled_frames_per_window"] = samples
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--samples", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = repo_root()
    tmpdir = Path(tempfile.mkdtemp(prefix="sampling_ablation_"))
    outputs = []
    try:
        for n in args.samples:
            cfg_path = tmpdir / f"sample{n}.yaml"
            write_sampling_config(args.config, n, cfg_path)
            manifest = project / "manifests" / f"local_windows_sample{n}.jsonl"
            emb = project / "features" / f"visual_mobilenet_v3_small_sample{n}.npz"
            mem = project / "retrieval" / f"memory_mobilenet_v3_small_sample{n}_mean_std.npz"
            run(["python", str(project / "scripts" / "build_window_manifest.py"), "--config", str(cfg_path), "--out", str(manifest)])
            run(["python", str(project / "scripts" / "extract_visual_embeddings.py"), "--config", str(cfg_path), "--manifest", str(manifest), "--out", str(emb), "--device", args.device])
            run(["python", str(project / "scripts" / "build_retrieval_memory.py"), "--config", str(cfg_path), "--embeddings", str(emb), "--out", str(mem), "--feature-mode", "mean_std"])
            run(["python", str(project / "scripts" / "evaluate_retrieval.py"), "--memory", str(mem)])
            outputs.append({"sampled_frames": n, "manifest": str(manifest), "embeddings": str(emb), "memory": str(mem)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(json.dumps({"sampling_ablation": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

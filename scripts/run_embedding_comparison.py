import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from pipeline_utils import ensure_dir, load_config, repo_root, resolve_path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def write_model_config(base_config: Path, model: str, out_path: Path) -> None:
    cfg = load_config(base_config)
    cfg["embeddings"]["visual_model"] = model
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--models", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = repo_root()
    project = base
    ensure_dir(project / "reports")
    if not args.skip_manifest:
        run(["python", str(project / "scripts" / "build_window_manifest.py"), "--config", str(args.config)])
    outputs = []
    tmpdir = Path(tempfile.mkdtemp(prefix="embedding_compare_"))
    try:
        for model in args.models:
            cfg_path = tmpdir / f"{model}.yaml"
            write_model_config(args.config, model, cfg_path)
            emb_out = project / "features" / f"visual_{model}_compare.npz"
            mem_out = project / "retrieval" / f"memory_{model}_mean_std_compare.npz"
            run(["python", str(project / "scripts" / "extract_visual_embeddings.py"), "--config", str(cfg_path), "--out", str(emb_out), "--device", args.device])
            run(["python", str(project / "scripts" / "build_retrieval_memory.py"), "--config", str(cfg_path), "--embeddings", str(emb_out), "--out", str(mem_out), "--feature-mode", "mean_std"])
            run(["python", str(project / "scripts" / "evaluate_retrieval.py"), "--memory", str(mem_out)])
            outputs.append({"model": model, "embeddings": str(emb_out), "memory": str(mem_out)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(json.dumps({"models": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

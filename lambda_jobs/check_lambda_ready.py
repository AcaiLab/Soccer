import json
import os
import shutil
import subprocess
from pathlib import Path


def command_version(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip().splitlines()[0]
    except Exception as exc:
        return f"error: {exc}"


def main() -> int:
    import torch
    import torchvision
    import pandas
    import cv2
    import yaml

    payload = {
        "cwd": str(Path.cwd()),
        "python_ok": True,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "pandas": pandas.__version__,
        "opencv": cv2.__version__,
        "yaml_ok": yaml is not None,
        "ffmpeg": command_version(["ffmpeg", "-version"]),
        "git": command_version(["git", "--version"]),
        "git_lfs": command_version(["git-lfs", "--version"]),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "disk": shutil.disk_usage(".")._asdict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

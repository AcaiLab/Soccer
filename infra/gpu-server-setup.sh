#!/bin/bash
# One-time setup on the GPU server, run as your normal user (no sudo needed).
#
#   bash infra/gpu-server-setup.sh            # CUDA 12.1 build of torch (driver >= 525)
#   bash infra/gpu-server-setup.sh cu118      # older drivers (>= 450)
#
# Creates .venv in the repo, installs pinned dependencies, checks the GPU, and
# writes a template ~/.config/soccer/env (mode 600) for your settings.
set -euo pipefail
CUDA_TAG="${1:-cu121}"
cd "$(dirname "$0")/.."

echo "== GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
else
  echo "nvidia-smi not found: is this the GPU machine?" >&2
  exit 1
fi

echo "== Python"
PY="$(command -v python3.11 || command -v python3)"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python 3.10+, got " + sys.version; print(sys.version)'

echo "== Virtual environment"
"$PY" -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install --quiet -r requirements.txt
pip install --quiet -e .
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

echo "== Settings file"
umask 077
mkdir -p "$HOME/.config/soccer"
if [ ! -f "$HOME/.config/soccer/env" ]; then
  cat > "$HOME/.config/soccer/env" << 'EOF'
# Load with:  source ~/.config/soccer/env      (this file is readable only by you)
export SOCCER_DATA_ROOT="$HOME/soccer-data"      # ~0.4 GB per game for frames, plus 1.2 GB while a game's video is local
export CUDA_VISIBLE_DEVICES=0                    # pick a free GPU (see nvidia-smi)

# Dropbox: the link and its password, and either a refresh token (does not expire, see the
# README) or a file holding a fresh 4-hour access token. Both files are mode 600.
export DROPBOX_SHARED_URL='REPLACE'
export DROPBOX_LINK_PASSWORD='REPLACE'
# export DROPBOX_OAUTH_FILE="$HOME/.config/soccer/dropbox_oauth.json"   # from scripts/dropbox_auth.py
export DROPBOX_TOKEN_FILE="$HOME/.config/soccer/dropbox_token"

# Optional, only if the data is staged in S3 (needs AWS access):
# export SOCCER_S3_ROOT=s3://bucket
# export AWS_PROFILE=soccer
# export AWS_DEFAULT_REGION=us-east-1
# export SOCCER_SSM_PREFIX=/soccer
EOF
  echo "Wrote $HOME/.config/soccer/env: edit the REPLACE values"
else
  echo "$HOME/.config/soccer/env already exists; left unchanged"
fi

echo "== Disk"
df -h "$HOME" | tail -1
echo "Setup complete. Next: edit ~/.config/soccer/env, then see docs/remote_gpu_server.md"

set -euo pipefail


echo "Python:"
python --version

echo "GPU:"
nvidia-smi || true

echo "System packages:"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg git git-lfs tmux htop rsync unzip aria2
fi

python -m pip install --upgrade pip
python -m pip install \
  numpy pandas scikit-learn pillow opencv-python pyyaml tqdm \
  torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  huggingface_hub hf_transfer open_clip_torch openai dropbox

echo "Environment check:"
python lambda_jobs/check_lambda_ready.py

echo "Lambda environment setup complete."

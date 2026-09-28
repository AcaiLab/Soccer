#!/bin/bash
set -euxo pipefail
exec > /var/log/soccer-ml-setup.log 2>&1

BUCKET="s3://soccer-ml-staging-910504055031"
WORK="/home/ubuntu/soccer"

# --- System updates ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq htop tmux tree jq ffmpeg

# --- Create workspace ---
mkdir -p $WORK/{results_by_label_epl_2021_2022,data/json,features,tmp_dbx,preprocess,clustering}
cd $WORK

# --- Pull code from S3 ---
aws s3 sync $BUCKET/code/ $WORK/ --exclude "preprocess/*"
aws s3 sync $BUCKET/code/preprocess/ $WORK/preprocess/
aws s3 sync $BUCKET/config/tmp_dbx/ $WORK/tmp_dbx/

# --- Install Python dependencies ---
pip install --upgrade pip
pip install transformers>=4.35.0 huggingface-hub>=0.19.0
pip install numpy pandas scikit-learn matplotlib Pillow opencv-python tqdm requests

# --- Verify GPU ---
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- Pull features and JSON (small, fast) ---
aws s3 sync $BUCKET/json/ $WORK/data/json/Soccer_Data_Json/ --only-show-errors
aws s3 sync $BUCKET/features/ $WORK/features/ --only-show-errors

# --- Pull frames from S3 (large, runs in background) ---
nohup aws s3 sync $BUCKET/frames/ $WORK/results_by_label_epl_2021_2022/ \
  --only-show-errors > /var/log/s3-frames-sync.log 2>&1 &

# --- Auto-shutdown script (idle protection) ---
cat > /home/ubuntu/auto-shutdown.sh << 'SHUTDOWN_SCRIPT'
#!/bin/bash
IDLE_THRESHOLD=30
IDLE_COUNT=0
while true; do
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$GPU_UTIL" ] || [ "$GPU_UTIL" -lt 5 ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
    else
        IDLE_COUNT=0
    fi
    if [ "$IDLE_COUNT" -ge "$IDLE_THRESHOLD" ]; then
        echo "$(date): GPU idle for ${IDLE_THRESHOLD} min. Shutting down." >> /var/log/auto-shutdown.log
        sudo shutdown -h now
    fi
    sleep 60
done
SHUTDOWN_SCRIPT
chmod +x /home/ubuntu/auto-shutdown.sh

# --- Cost check helper ---
cat > /home/ubuntu/cost-check.sh << 'COST_SCRIPT'
#!/bin/bash
UPTIME_HOURS=$(awk '{printf "%.1f", $1/3600}' /proc/uptime)
COST=$(python3 -c "print(f'{float(${UPTIME_HOURS}) * 0.526:.2f}')")
echo "Uptime: ${UPTIME_HOURS}h | Est. cost: \$${COST}"
COST_SCRIPT
chmod +x /home/ubuntu/cost-check.sh

# --- Ownership ---
chown -R ubuntu:ubuntu $WORK /home/ubuntu/*.sh

echo "=== SETUP COMPLETE $(date) ==="

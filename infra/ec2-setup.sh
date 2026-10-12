#!/bin/bash
# GPU instance bootstrap (EC2 user data) for the feature / training stages.
#
# Required:  SOCCER_BUCKET      s3://bucket that holds code/, json/, frames/, features/
# Optional:  SOCCER_SSM_PREFIX  SSM path holding the Dropbox settings (default /soccer)
#
# Credentials: the instance role grants S3 and SSM access. No tokens are copied
# onto the instance; soccer/dropbox.py reads SSM SecureStrings when needed.
set -euxo pipefail
exec > /var/log/soccer-ml-setup.log 2>&1

: "${SOCCER_BUCKET:?Set SOCCER_BUCKET, e.g. s3://soccer-ml-staging-<account>}"
SSM_PREFIX="${SOCCER_SSM_PREFIX:-/soccer}"
WORK="/home/ubuntu/soccer"

# --- System packages ---
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq htop tmux tree jq ffmpeg

# --- Code and pinned dependencies ---
mkdir -p "$WORK"
cd "$WORK"
aws s3 sync "$SOCCER_BUCKET/code/" "$WORK/" --only-show-errors
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# --- Environment for every login shell ---
cat > /etc/profile.d/soccer.sh << EOF
export SOCCER_DATA_ROOT=$WORK
export SOCCER_SSM_PREFIX=$SSM_PREFIX
EOF

# --- Verify GPU ---
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# --- Annotations and features (small, fast) ---
mkdir -p "$WORK/data/json/Soccer_Data_Json" "$WORK/features" "$WORK/frames_v2"
aws s3 sync "$SOCCER_BUCKET/json/" "$WORK/data/json/Soccer_Data_Json/" --only-show-errors
aws s3 sync "$SOCCER_BUCKET/features/" "$WORK/features/" --only-show-errors

# --- Frames (large, in the background) ---
nohup aws s3 sync "$SOCCER_BUCKET/frames/" "$WORK/frames_v2/" \
  --only-show-errors > /var/log/s3-frames-sync.log 2>&1 &

# --- Auto-shutdown after 30 idle GPU minutes ---
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
# Not started automatically: the GPU is idle while frames sync, so it would stop
# the instance early. Start it once work is running: nohup ~/auto-shutdown.sh &

# --- Cost check helper (edit HOURLY_USD for the instance type) ---
cat > /home/ubuntu/cost-check.sh << 'COST_SCRIPT'
#!/bin/bash
HOURLY_USD=0.526
UPTIME_HOURS=$(awk '{printf "%.1f", $1/3600}' /proc/uptime)
echo "Uptime: ${UPTIME_HOURS}h | Est. cost: \$$(awk -v h="$UPTIME_HOURS" -v r="$HOURLY_USD" 'BEGIN {printf "%.2f", h*r}')"
COST_SCRIPT
chmod +x /home/ubuntu/cost-check.sh

chown -R ubuntu:ubuntu "$WORK" /home/ubuntu/*.sh
echo "=== SETUP COMPLETE $(date) ==="

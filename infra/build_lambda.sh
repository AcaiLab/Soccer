#!/bin/bash
# Build the Dropbox -> S3 transfer Lambda package (infra/build/transfer_lambda.zip).
# The Lambda runtime already ships boto3; only requests and the soccer package
# are bundled. Run from the repository root on Linux or in a Linux container
# (Lambda runs Linux).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/infra/build"
rm -rf "$build/pkg" && mkdir -p "$build/pkg"
pip install --quiet --target "$build/pkg" requests==2.32.3 pyyaml==6.0.2
cp -r "$root/soccer" "$build/pkg/soccer"
find "$build/pkg" -name "__pycache__" -prune -exec rm -rf {} +
(cd "$build/pkg" && zip -qr "$build/transfer_lambda.zip" .)
echo "Built $build/transfer_lambda.zip"

set -euo pipefail


MANIFEST="${MANIFEST:-/workspace/data/SoccerReplay-1988/video_urls.txt}"
OUT_DIR="${OUT_DIR:-/workspace/data/SoccerReplay-1988/videos}"
TOOL="${TOOL:-aria2c}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Video URL manifest not found: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

if [[ "$TOOL" == "aria2c" ]]; then
  if ! command -v aria2c >/dev/null 2>&1; then
    echo "aria2c not found; install it or set TOOL=wget" >&2
    exit 2
  fi
  grep -v '^\s*$' "$MANIFEST" | grep -v '^\s*#' > "$OUT_DIR/video_urls.clean.txt"
  aria2c -x "${ARIA_CONNECTIONS:-8}" -s "${ARIA_SPLITS:-8}" -c -d "$OUT_DIR" -i "$OUT_DIR/video_urls.clean.txt"
elif [[ "$TOOL" == "wget" ]]; then
  while IFS= read -r url; do
    [[ -z "$url" || "$url" =~ ^[[:space:]]*# ]] && continue
    wget -c -P "$OUT_DIR" "$url"
  done < "$MANIFEST"
else
  echo "Unknown TOOL=$TOOL; use aria2c or wget" >&2
  exit 2
fi

find "$OUT_DIR" -type f | sort > "$OUT_DIR/downloaded_files.txt"
echo "Downloaded files listed at $OUT_DIR/downloaded_files.txt"

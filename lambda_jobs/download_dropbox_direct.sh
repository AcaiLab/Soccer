set -euo pipefail


URL="${DROPBOX_SHARED_LINK:-${1:-}}"
OUT="${OUT:-${2:-/workspace/data/dropbox_download.zip}}"

if [[ -z "$URL" ]]; then
  echo "Usage: DROPBOX_SHARED_LINK='https://www.dropbox.com/...' OUT=/path/file.zip $0" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

if [[ "$URL" == *"?"* ]]; then
  URL="${URL/dl=0/dl=1}"
  if [[ "$URL" != *"dl=1"* ]]; then
    URL="${URL}&dl=1"
  fi
else
  URL="${URL}?dl=1"
fi

echo "Downloading Dropbox shared link to $OUT"
curl -L --fail --retry 5 --retry-delay 5 -o "$OUT" "$URL"
ls -lh "$OUT"

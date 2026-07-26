#!/bin/sh
# Maintainer helper: download a contribution bundle from the link posted in a
# 'Shard contribution' issue, resolving the common file hosts to their direct
# download URL. Resumes (-C -) and retries, then verifies the gzip.
#
# Usage: ./scripts/fetch_contribution.sh <issue-link> <output.tar.gz>
#
# Resolvable hosts:
#   Dropbox           www.dropbox.com/scl/...?...&dl=0   -> forces &dl=1
#   Google Drive      /file/d/<id>/view  or  ?id=<id>    -> usercontent direct
#   Nextcloud share   <host>/s/<token>                   -> appends /download
#   anything else     used verbatim (already a direct link)
#
# Hosts that CANNOT be scripted (JavaScript/token/E2E-encrypted) — download
# these by hand in a browser and pass the local file to ingest_contribution.py:
#   swisstransfer.com, transfert.free.fr, web.tresorit.com
set -eu

URL="${1:?usage: fetch_contribution.sh <issue-link> <output.tar.gz>}"
OUT="${2:?usage: fetch_contribution.sh <issue-link> <output.tar.gz>}"

case "$URL" in
  *dropbox.com*)
    DIRECT=$(printf '%s' "$URL" | sed -E 's/([?&])dl=0/\1dl=1/');
    case "$DIRECT" in *dl=1*) : ;; *) DIRECT="$DIRECT&dl=1" ;; esac ;;
  *drive.google.com/file/d/*)
    ID=$(printf '%s' "$URL" | sed -E 's#.*/file/d/([^/]+)/.*#\1#')
    DIRECT="https://drive.usercontent.google.com/download?export=download&confirm=t&id=$ID" ;;
  *drive.google.com*id=*|*drive.usercontent.google.com*)
    ID=$(printf '%s' "$URL" | sed -E 's#.*[?&]id=([^&]+).*#\1#')
    DIRECT="https://drive.usercontent.google.com/download?export=download&confirm=t&id=$ID" ;;
  *swisstransfer.com*|*transfert.free.fr*|*tresorit.com*)
    echo "ERROR: $URL is on a host that cannot be scripted (JS/token/E2E)." >&2
    echo "Download it in a browser and pass the local file to ingest_contribution.py." >&2
    exit 2 ;;
  */s/*)  # Nextcloud / ownCloud public share
    DIRECT="${URL%/}/download" ;;
  *)
    DIRECT="$URL" ;;
esac

echo "resolved -> $DIRECT"
curl -fL -C - --retry 10 --retry-delay 5 --retry-all-errors -A "Mozilla/5.0" "$DIRECT" -o "$OUT"
if gzip -t "$OUT" 2>/dev/null; then
    echo "OK: $OUT ($(du -h "$OUT" | cut -f1), sha256 $(sha256sum "$OUT" | cut -c1-16)...)"
else
    echo "ERROR: $OUT is not a valid gzip — the host likely served an HTML page." >&2
    echo "Open the link in a browser to check, or download by hand." >&2
    exit 1
fi

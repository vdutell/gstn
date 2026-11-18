#!/usr/bin/env bash
# Download and unpack the asset bundle (syntheses, targets, video frames,
# compressor checkpoints). About 372 MB compressed, 409 MB on disk.
#
#   bash data/fetch_assets.sh
#
# Override the destination with GSTN_ASSETS, and the source with GSTN_ASSET_URL.
set -euo pipefail

VERSION="v1"
URL="${GSTN_ASSET_URL:-https://github.com/vdutell/gstn/releases/download/assets-${VERSION}/gstn-assets-${VERSION}.tar.gz}"
SHA256="3f8a9062602467a1d810bf28e5b351843c4e88406e51f256de5e23bccd61a62c"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${GSTN_ASSETS:-$HERE/assets}"
TARBALL="$HERE/gstn-assets-${VERSION}.tar.gz"

if [ -d "$DEST/checkpoints" ] && [ -d "$DEST/synths" ]; then
  echo "Assets already present at $DEST"
  echo "Delete that directory to re-download."
  exit 0
fi

echo "Downloading $URL"
mkdir -p "$HERE"
if ! curl -fL --retry 3 -o "$TARBALL" "$URL"; then
  cat >&2 <<'MSG'

ERROR: download failed.

If the release has moved, point at it directly:
    GSTN_ASSET_URL=https://.../gstn-assets-v1.tar.gz bash data/fetch_assets.sh

If you already have the bundle, or built it with data/make_asset_bundle.sh,
skip the download entirely:
    export GSTN_ASSETS=/path/to/gstn-assets
MSG
  rm -f "$TARBALL"
  exit 1
fi

echo "Verifying checksum"
echo "${SHA256}  ${TARBALL}" | sha256sum -c -

echo "Unpacking into $DEST"
mkdir -p "$DEST"
tar xzf "$TARBALL" -C "$DEST"
rm -f "$TARBALL"

echo
echo "Done. Contents:"
du -sh "$DEST"/* | sort -rh

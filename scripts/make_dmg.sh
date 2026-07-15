#!/usr/bin/env bash
# Wraps the (signed) .app in a drag-to-Applications .dmg for distribution.
# Run after build_app.sh (and sign_and_notarize.sh, if distributing to others).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BUNDLE="$REPO_ROOT/dist/Ifrit3D-MLX.app"
DMG_PATH="$REPO_ROOT/dist/Ifrit3D-MLX.dmg"
STAGING_DIR="$REPO_ROOT/dist/.dmg_staging"

if [ ! -d "$APP_BUNDLE" ]; then
  echo "ERROR: $APP_BUNDLE not found — run scripts/build_app.sh first" >&2
  exit 1
fi

echo "==> Staging DMG contents"
rm -rf "$STAGING_DIR" "$DMG_PATH"
mkdir -p "$STAGING_DIR"
cp -R "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

echo "==> Building $DMG_PATH"
hdiutil create -volname "Ifrit3D-MLX" \
  -srcfolder "$STAGING_DIR" \
  -ov -format UDZO \
  "$DMG_PATH"

rm -rf "$STAGING_DIR"

echo "==> Done: $DMG_PATH"

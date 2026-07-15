#!/usr/bin/env bash
# Signs and notarizes the .app built by build_app.sh. This is the one step
# that needs your own Apple Developer credentials — see the plan/README for
# the one-time setup (Developer ID certificate + `xcrun notarytool
# store-credentials`). This script never touches raw credentials, only a
# certificate identity string and a keychain profile name you've already
# created yourself.
#
# Usage:
#   scripts/sign_and_notarize.sh "Developer ID Application: NAME (TEAMID)" <notarytool-profile>
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 \"Developer ID Application: NAME (TEAMID)\" <notarytool-profile>" >&2
  exit 1
fi

SIGN_IDENTITY="$1"
NOTARY_PROFILE="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BUNDLE="$REPO_ROOT/dist/Ifrit3D-MLX.app"
ENTITLEMENTS="$REPO_ROOT/scripts/entitlements.plist"
ZIP_PATH="$REPO_ROOT/dist/Ifrit3D-MLX.zip"

if [ ! -d "$APP_BUNDLE" ]; then
  echo "ERROR: $APP_BUNDLE not found — run scripts/build_app.sh first" >&2
  exit 1
fi

echo "==> Signing every Mach-O binary inside the bundle"
# Every executable/dylib/so needs its own signature under hardened runtime —
# codesign --deep on the outer bundle alone is unreliable for a tree this
# deep (torch, mlx, and co. ship many nested dylibs). Enumerate explicitly.
while IFS= read -r -d '' f; do
  if file "$f" | grep -q "Mach-O"; then
    codesign --force --timestamp --options runtime \
      --entitlements "$ENTITLEMENTS" \
      --sign "$SIGN_IDENTITY" \
      "$f"
  fi
done < <(find "$APP_BUNDLE" -type f \( -perm -u+x -o -name "*.so" -o -name "*.dylib" \) -print0)

echo "==> Signing the outer app bundle"
codesign --force --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$SIGN_IDENTITY" \
  "$APP_BUNDLE"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"

echo "==> Zipping for notarization"
rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"

echo "==> Submitting to Apple notary service (this can take several minutes)"
xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling the notarization ticket"
xcrun stapler staple "$APP_BUNDLE"

echo "==> Final check"
spctl -a -vv "$APP_BUNDLE"
xcrun stapler validate "$APP_BUNDLE"

echo "==> Done: $APP_BUNDLE is signed and notarized"

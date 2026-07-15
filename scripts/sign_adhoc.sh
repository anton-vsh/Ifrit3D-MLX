#!/usr/bin/env bash
# Ad-hoc signs the .app built by build_app.sh. No Apple Developer account
# needed — this is the "ship it as-is" path, not the notarized one.
#
# Apple Silicon's kernel refuses to execute a Mach-O binary with *no*
# signature at all, so an entirely unsigned .app downloaded from the
# internet shows Gatekeeper's scariest error: "Ifrit3D-MLX is damaged
# and can't be opened. You should move it to the Trash." Ad-hoc signing
# (identity "-", no certificate) satisfies that kernel-level requirement
# without needing a Developer ID. Recipients still see one normal
# "unidentified developer" warning on first launch — that's expected and
# is bypassed with right-click -> Open (see the instructions this script
# prints at the end).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BUNDLE="$REPO_ROOT/dist/Ifrit3D-MLX.app"
ENTITLEMENTS="$REPO_ROOT/scripts/entitlements.plist"

if [ ! -d "$APP_BUNDLE" ]; then
  echo "ERROR: $APP_BUNDLE not found — run scripts/build_app.sh first" >&2
  exit 1
fi

echo "==> Ad-hoc signing every Mach-O binary inside the bundle"
while IFS= read -r -d '' f; do
  if file "$f" | grep -q "Mach-O"; then
    codesign --force --sign - --entitlements "$ENTITLEMENTS" "$f"
  fi
done < <(find "$APP_BUNDLE" -type f \( -perm -u+x -o -name "*.so" -o -name "*.dylib" \) -print0)

echo "==> Ad-hoc signing the outer app bundle"
codesign --force --sign - --entitlements "$ENTITLEMENTS" "$APP_BUNDLE"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP_BUNDLE"

echo ""
echo "==> Done: $APP_BUNDLE is ad-hoc signed."
echo ""
echo "This is NOT notarized — 'spctl -a -vv' will report it as rejected by"
echo "Gatekeeper's source policy, which is expected without a Developer ID."
echo "Recipients need to bypass Gatekeeper's warning ONCE on first launch:"
echo "  1. Right-click (or Control-click) Ifrit3D-MLX.app -> Open"
echo "  2. Click \"Open\" in the dialog that appears"
echo "After that first approval, macOS remembers it and launches normally"
echo "(including double-click) from then on."
